# How to Start Hosting LLMs on OpenRouter (vLLM / SGLang)

Practical starter guide (realistic, not hype).

Becoming a successful OpenRouter provider with one rented GPU is hard — there’s a big application backlog, they prefer proprietary models, and traffic goes to the fastest/cheapest/most reliable endpoints. Treat this first as a learning + personal-use project. If it works and you get traffic, scale later.

---

## 1. Which Cloud to Rent

| Priority          | Platform     | Why                                      | Best for starting                  | Notes                          |
|-------------------|--------------|------------------------------------------|------------------------------------|--------------------------------|
| **Best balance**  | **RunPod**   | Easy UI, good templates for vLLM/SGLang, Community Cloud is cheap, Secure Cloud more reliable | Most people                       | Use Community first            |
| Cheapest          | **Vast.ai**  | Often lowest prices (marketplace)        | Pure experimentation              | Variable quality/uptime        |
| More reliable     | Lambda / others | Better SLAs                           | Later if you get real traffic     | More expensive                 |

### Recommended Starting GPU (prices fluctuate)

- **RTX 4090 / 5090 (24–32 GB)** → cheapest good option (~$0.30–0.50/hr on RunPod Community / Vast)
- Better: **L40S / A6000 / RTX PRO 6000 (48 GB)** or **A100 40/80 GB** if budget allows
- Avoid tiny cards (<24 GB) if you want anything useful

Start with **1× 24–48 GB** card. You can always stop the pod when not testing.

---

## 2. Which Model to Run First

Current high-demand open models on OpenRouter (token volume leaders) are mostly Chinese flash/MoE models: **DeepSeek V4.x Flash**, **GLM 5.3 Flash**, **MiMo-V2.6-Flash**, Qwen variants, Nemotron, Kimi, etc.

### Realistic single-GPU starters (fit + demand)

- **Best first try**: Qwen3 / Qwen2.5 14B–32B (or similar mid-size dense) in AWQ/GPTQ/FP8
- Strong MoE options that fit 24–48 GB when quantized (check exact VRAM with the engine)
- Smaller high-quality: 7B–14B class if you want higher concurrency
- Avoid giant models (70B+ dense full precision) on one consumer card

**Rule**: Pick a model that already has decent volume on OpenRouter *and* that you can serve at competitive price + good latency.  
Check live rankings at [openrouter.ai/rankings](https://openrouter.ai/rankings) and individual model pages for current providers/prices.

---

## 3. Quick Start Steps

1. **Create account** on RunPod (or Vast.ai) → rent a pod with the GPU above.  
   Use an official vLLM or SGLang template if available, or a clean Ubuntu + CUDA image.

2. **Install & run the server** (example with vLLM — SGLang is very similar):

```bash
pip install vllm   # or the latest recommended version

vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --api-key YOUR_SECRET_KEY
