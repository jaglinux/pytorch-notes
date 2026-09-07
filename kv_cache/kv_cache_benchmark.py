import time
import importlib.util
import torch

# Incompatible torchvision can crash on import:
#   RuntimeError: operator torchvision::nms does not exist
# Qwen2.5-Instruct is text-only, so hide torchvision before transformers loads.
_orig_find_spec = importlib.util.find_spec


def _find_spec(name, package=None):
    if name == "torchvision" or (isinstance(name, str) and name.startswith("torchvision.")):
        return None
    return _orig_find_spec(name, package)


importlib.util.find_spec = _find_spec

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

amd_gpu_detected = bool(getattr(torch.version, "hip", None))
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    amd_gpu_detected = amd_gpu_detected or any(
        token in device_name.lower() for token in ("amd", "radeon", "instinct")
    )
    print(f"GPU detected: {device_name} (AMD: {amd_gpu_detected})")
else:
    print(f"No CUDA/ROCm GPU detected (AMD: {amd_gpu_detected})")

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"Fetching '{model_id}' via Hugging Face...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
device = model.device
model.eval()
print(f"Model placed on: {device} (of {torch.cuda.device_count()} visible GPUs)")

# Per-step latency has a ~11 ms eager-mode floor here, so short contexts hide the
# KV-cache win entirely. Recompute only exceeds that floor past a few thousand tokens.
target_ctx = 16384
base = (
    "Explain computing, memory, and transformers in detail. "
    "Cover GPUs, attention, KV cache, latency, and throughput. "
)
repeats = -(-target_ctx // len(tokenizer(base).input_ids))
prompt_string = base * repeats
inputs = tokenizer(prompt_string, return_tensors="pt").to(device)
input_ids = inputs.input_ids[:, :target_ctx]
prompt_len = input_ids.shape[1]
max_new_tokens = 16
warmup_steps = 4

prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
print("\n================ PROMPT ================")
print("The 15k length is NOT 15 unique sentences. The same `base` string is")
print("repeated in Python with `base * repeats`, then cut to target_ctx tokens.")
print(f"base text              : {base!r}")
print(f"base token count       : {len(tokenizer(base).input_ids)}")
print(f"repeats                : {repeats}")
print(f"chars before tokenize  : {len(prompt_string)}")
print(f"Prompt length          : {prompt_len} tokens")
print(f"Tokens to generate     : {max_new_tokens}  (table rows 0 through {max_new_tokens - 1})")
print("--- full prompt text ---")
print(prompt_text)
print("=======================================")


def sync():
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def mem_mb(nbytes):
    return nbytes / (1024 ** 2)


# Memory and sync calls must use the model's device. The no-arg forms
# track the default device, which can differ from where the model was placed.
def allocated():
    return torch.cuda.memory_allocated(device) if device.type == "cuda" else 0


def print_cuda_memory(label):
    print(f"\n--- CUDA memory ({device}): {label} ---")
    if device.type != "cuda":
        print("  (no CUDA/ROCm device)")
        return
    sync()
    print(f"  allocated         : {mem_mb(torch.cuda.memory_allocated(device)):10.2f} MB")
    print(f"  reserved          : {mem_mb(torch.cuda.memory_reserved(device)):10.2f} MB")
    print(f"  max allocated     : {mem_mb(torch.cuda.max_memory_allocated(device)):10.2f} MB")
    print(f"  max reserved      : {mem_mb(torch.cuda.max_memory_reserved(device)):10.2f} MB")
    free_b, total_b = torch.cuda.mem_get_info(device)
    print(f"  device free/total : {mem_mb(free_b):10.2f} / {mem_mb(total_b):.2f} MB")


def reset_cuda_peak():
    if device.type == "cuda":
        sync()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.reset_accumulated_memory_stats(device)


def kv_cache_stats(cache):
    tensors = []
    for name in ("key_cache", "value_cache", "layers"):
        if hasattr(cache, name):
            attr = getattr(cache, name)
            if attr is None:
                continue
            if isinstance(attr, torch.Tensor):
                tensors.append(attr)
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, torch.Tensor):
                        tensors.append(item)
                    elif hasattr(item, "keys") and hasattr(item, "values"):
                        k, v = item.keys, item.values
                        if isinstance(k, torch.Tensor):
                            tensors.append(k)
                        if isinstance(v, torch.Tensor):
                            tensors.append(v)
    total_bytes = sum(t.numel() * t.element_size() for t in tensors)
    seq_len = 0
    try:
        seq_len = int(cache.get_seq_length())
    except Exception:
        pass
    return total_bytes, len(tensors), seq_len


def next_token_id(logits):
    return torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)


def token_str(token_tensor):
    tid = int(token_tensor.item())
    text = tokenizer.decode([tid], skip_special_tokens=False)
    return tid, text


print_cuda_memory("after model load")

# Warm up both kernel shapes so compile/autotune is not in the timed table.
print(f"\n[Warmup] prompt={prompt_len} tokens, {warmup_steps} steps each path...")
with torch.no_grad():
    warm = input_ids.clone()
    for _ in range(warmup_steps):
        out = model(warm, use_cache=False)
        warm = torch.cat([warm, next_token_id(out.logits)], dim=1)
    cache = DynamicCache()
    warm = input_ids.clone()
    pos = torch.arange(prompt_len, device=device)
    out = model(warm, cache_position=pos, past_key_values=cache, use_cache=True)
    tok = next_token_id(out.logits)
    for _ in range(warmup_steps - 1):
        pos = torch.tensor([cache.get_seq_length()], device=device)
        out = model(tok, cache_position=pos, past_key_values=cache, use_cache=True)
        tok = next_token_id(out.logits)
sync()
del cache, warm, out, tok
print_cuda_memory("after warmup")

# =========================================================================
# EXPERIMENT 1: PURE RECOMPUTATION (NO KV CACHE)
# =========================================================================
print("\n[Running Experiment 1: Standard Inference Without Cache]")
reset_cuda_peak()
mem_before_nocache = allocated()
nocache_latencies = []
nocache_pred_ids = []
nocache_input = input_ids.clone()

for step in range(max_new_tokens):
    sync()
    t_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(nocache_input, use_cache=False)
    sync()
    nocache_latencies.append((time.perf_counter() - t_start) * 1000)
    tok = next_token_id(outputs.logits)
    nocache_pred_ids.append(tok)
    nocache_input = torch.cat([nocache_input, tok], dim=1)

sync()
mem_after_nocache = allocated()
peak_nocache = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
print("\n--- Experiment 1 predicted tokens ---")
print("step | token id | token text")
for i, tok in enumerate(nocache_pred_ids):
    tid, text = token_str(tok)
    print(f" {i:4d} | {tid:8d} | {text!r}")
print(" decoded continuation:", tokenizer.decode([int(t.item()) for t in nocache_pred_ids]))
print_cuda_memory("experiment 1 (no cache) done")

# Free the no-cache logits (prompt_len x vocab, several GB) so experiment 2's
# baseline is not measured against experiment 1's leftovers.
del outputs, nocache_input
if device.type == "cuda":
    torch.cuda.empty_cache()

# =========================================================================
# EXPERIMENT 2: OPTIMIZED PROCESSING (WITH DYNAMIC CACHE)
# =========================================================================
print("\n[Running Experiment 2: Optimized Inference With DynamicCache]")
reset_cuda_peak()
mem_before_cache = allocated()
cache_latencies = []
cache_pred_ids = []
kv_cache = DynamicCache()
current_input = input_ids.clone()

for step in range(max_new_tokens):
    sync()
    t_start = time.perf_counter()
    with torch.no_grad():
        if step == 0:
            cache_position = torch.arange(prompt_len, device=device)
            outputs = model(
                current_input,
                cache_position=cache_position,
                past_key_values=kv_cache,
                use_cache=True,
            )
        else:
            cache_position = torch.tensor([kv_cache.get_seq_length()], device=device)
            outputs = model(
                current_input,
                cache_position=cache_position,
                past_key_values=kv_cache,
                use_cache=True,
            )
    sync()
    cache_latencies.append((time.perf_counter() - t_start) * 1000)
    current_input = next_token_id(outputs.logits)
    cache_pred_ids.append(current_input)

sync()
kv_bytes, kv_tensors, kv_seq = kv_cache_stats(kv_cache)
print("\n--- Experiment 2 predicted tokens ---")
print("step | token id | token text")
for i, tok in enumerate(cache_pred_ids):
    tid, text = token_str(tok)
    print(f" {i:4d} | {tid:8d} | {text!r}")
print(" decoded continuation:", tokenizer.decode([int(t.item()) for t in cache_pred_ids]))
mem_after_cache = allocated()
peak_cache = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
print_cuda_memory("experiment 2 (with KV cache) done")
print("\n--- KV cache object ---")
print(f"  sequence length    : {kv_seq} tokens")
print(f"  stored tensors     : {kv_tensors}")
print(f"  KV bytes in RAM/VRAM: {mem_mb(kv_bytes):10.2f} MB")
print(f"  allocated delta    : {mem_mb(mem_after_cache - mem_before_cache):10.2f} MB")

# =========================================================================
# PERFORMANCE GRAPH METRICS
# =========================================================================
n = min(len(nocache_latencies), len(cache_latencies))
print("\n================ BENCHMARK STATS COMPARED ================")
print(f"Prompt Size            : {prompt_len} tokens")
print(f"Generated Steps        : {n} steps")
print("----------------------------------------------------------")
print("Step | No-Cache Latency (ms) | Cached Latency (ms) | Speedup Factor")
print("----------------------------------------------------------")

for i in range(n):
    ratio = nocache_latencies[i] / cache_latencies[i]
    phase_label = f"{i} (Prefill)" if i == 0 else f"{i} (Decode) "
    print(f" {phase_label} | {nocache_latencies[i]:20.2f} | {cache_latencies[i]:19.2f} | {ratio:.2f}x")

decode_n = n - 1
if decode_n > 0:
    avg_nc = sum(nocache_latencies[1:n]) / decode_n
    avg_c = sum(cache_latencies[1:n]) / decode_n
    print("----------------------------------------------------------")
    print(f" Prefill (step 0)      : no-cache {nocache_latencies[0]:.2f} ms | cache {cache_latencies[0]:.2f} ms")
    print(f" Avg decode (steps 1+) : no-cache {avg_nc:.2f} ms | cache {avg_c:.2f} ms | {avg_nc / avg_c:.2f}x")
    print(" KV cache is helping if avg decode speedup is clearly above 1.0x")
print("==========================================================")
print("\n================ CUDA MEMORY SUMMARY ================")
print(f" No-cache peak allocated : {mem_mb(peak_nocache):10.2f} MB")
print(f" Cached peak allocated   : {mem_mb(peak_cache):10.2f} MB")
print(f" No-cache alloc delta    : {mem_mb(mem_after_nocache - mem_before_nocache):10.2f} MB")
print(f" Cached alloc delta      : {mem_mb(mem_after_cache - mem_before_cache):10.2f} MB")
print(f" KV cache stored         : {mem_mb(kv_bytes):10.2f} MB")
print("====================================================")
