# Running Your Own LLM Endpoint (No OpenRouter Required)

This guide is for developers who want to rent a GPU, host an open-source LLM, and use it personally through an OpenAI-compatible API.

**Goal:** Learn the stack, use the model from your own tools, and avoid the complexity of becoming a public provider.

---

# Architecture

```text
Your Laptop
    ↓
OpenAI SDK / Browser / IDE
    ↓
Your Private Endpoint
    ↓
vLLM or SGLang
    ↓
GPU (RunPod / Vast.ai / Local Machine)
    ↓
Open Source Model
```

You become your own AI provider.

---

# Why Start This Way

Advantages:

- No OpenRouter approval required
- No customers to support
- No uptime requirements
- No billing setup
- No privacy policy needed
- Learn the entire serving stack
- Use your endpoint immediately

This is the best way to determine whether running inference is interesting before worrying about monetization.

---

# Step 1: Rent a GPU

Recommended providers:

| Provider | Notes |
|-----------|---------|
| RunPod | Easiest for beginners |
| Vast.ai | Often cheapest |
| Lambda | More reliable but pricier |

### Recommended GPUs

| GPU | VRAM | Good For |
|------|------|----------|
| RTX 4090 | 24 GB | Cheapest practical option |
| RTX 5090 | 32 GB | Better headroom |
| RX 7900 XTX | 24 GB | ROCm experimentation |
| W7900 | 48 GB | Larger models |
| L40S | 48 GB | Excellent inference card |

Start with a single GPU.

---

# Step 2: Launch a Pod

Choose:

- Ubuntu + CUDA (NVIDIA)
- Ubuntu + ROCm (AMD)
- Prebuilt vLLM image
- Prebuilt SGLang image

Recommended for beginners:

```text
RunPod
    →
vLLM Template
```

This removes most setup work.

---

# Step 3: Install vLLM

Example:

```bash
pip install vllm
```

Verify:

```bash
python -c "import vllm; print(vllm.__version__)"
```

---

# Step 4: Download and Serve a Model

Example:

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct
```

More realistic production settings:

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --api-key mysecretkey
```

After startup:

```text
http://YOUR_SERVER:8000
```

or

```text
https://YOUR_RUNPOD_URL
```

will expose an OpenAI-compatible endpoint.

---

# Step 5: Verify the Endpoint

List models:

```bash
curl http://localhost:8000/v1/models
```

Expected response:

```json
{
  "data": [
    {
      "id": "Qwen2.5-14B-Instruct"
    }
  ]
}
```

---

# Step 6: Consume from Python

Install OpenAI client:

```bash
pip install openai
```

Example:

```python
from openai import OpenAI

client = OpenAI(
    api_key="mysecretkey",
    base_url="https://YOUR_ENDPOINT/v1"
)

response = client.chat.completions.create(
    model="Qwen2.5-14B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Explain grouped GEMM."
        }
    ]
)

print(response.choices[0].message.content)
```

---

# Step 7: Use from Open WebUI

Open WebUI provides a ChatGPT-like interface.

Run:

```bash
docker run -d \
  -p 3000:8080 \
  ghcr.io/open-webui/open-webui:main
```

Configure:

```text
Settings
    →
Connections
    →
OpenAI Compatible API
```

Point it to:

```text
https://YOUR_ENDPOINT/v1
```

Result:

```text
Browser
    ↓
Open WebUI
    ↓
Your Model
```

No OpenAI subscription required.

---

# Step 8: Use from VS Code

Many tools support OpenAI-compatible APIs.

Examples:

- Continue.dev
- Cline
- OpenCode
- Aider
- Roo Code

Configure:

```text
Base URL:
https://YOUR_ENDPOINT/v1

API Key:
mysecretkey
```

Now your IDE can use your hosted model.

---

# Step 9: Secure the Endpoint

Minimum security:

```text
Internet
   ↓
HTTPS
   ↓
API Key
   ↓
vLLM
```

Better options:

- Cloudflare Tunnel
- Tailscale
- WireGuard
- Private VPN

For personal use, API-key protection is usually sufficient.

---

# Step 10: Monitor Performance

Useful metrics:

| Metric | Meaning |
|----------|---------|
| TTFT | Time to first token |
| TPS | Tokens per second |
| VRAM | Memory usage |
| Throughput | Concurrent request capacity |
| Cost/hr | GPU rental cost |

Track:

```bash
nvidia-smi
```

or for AMD:

```bash
rocm-smi
```

---

# Suggested First Models

### Small and Fast

```text
Qwen2.5-7B-Instruct
Llama-3.1-8B-Instruct
Gemma-3-12B
```

### Strong Quality

```text
Qwen2.5-14B-Instruct
Qwen3-14B
Qwen3-32B-AWQ
```

### Larger Experiments

```text
Qwen3-32B
DeepSeek Distill Models
Nemotron Variants
```

---

# Estimated Monthly Cost

Assuming you run continuously:

| GPU | Hourly | Monthly (24x7) |
|-------|---------|---------|
| RTX 4090 | ~$0.30-$0.50 | ~$220-$365 |
| RTX 5090 | ~$0.40-$0.70 | ~$290-$510 |
| W7900 | Varies | Varies |
| L40S | ~$0.80-$1.50 | ~$580-$1,095 |

For learning:

```text
Run only when needed.
```

A few hours per week can keep costs below $20-$50/month.

---

# Example Personal Workflow

```text
RunPod RTX 4090
        ↓
vLLM
        ↓
Qwen2.5-14B
        ↓
OpenAI-compatible API
        ↓
VS Code + Continue
        ↓
Personal coding assistant
```

This gives you hands-on experience with:

- Model serving
- GPU utilization
- Quantization
- API hosting
- Inference performance
- Cost optimization

without needing OpenRouter, customers, or production obligations.

---

# Future Growth Path

```text
Single GPU
    ↓
Personal Usage
    ↓
Friends/Small Team
    ↓
Multiple GPUs
    ↓
Public Endpoint
    ↓
OpenRouter Provider
```

Start by serving yourself. If you find the setup reliable and useful, scaling and applying to OpenRouter becomes much easier.
