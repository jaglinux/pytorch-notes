# Becoming an OpenRouter Provider with One Rented GPU

Becoming a successful OpenRouter provider with a single rented GPU is challenging. There is a significant application backlog, OpenRouter tends to prioritize proprietary models, and traffic is generally routed to the fastest, cheapest, and most reliable endpoints.

Treat this primarily as a **learning and personal-use project**. If you gain traffic and prove reliability, you can scale later.

---

# Which Cloud to Rent

| Priority | Platform | Why | Best For | Notes |
|-----------|-----------|------|----------|--------|
| Best balance | RunPod | Easy UI, good templates for vLLM/SGLang, Community Cloud is cheap, Secure Cloud is more reliable | Most people | Start with Community Cloud |
| Cheapest | Vast.ai | Often the lowest prices due to marketplace model | Pure experimentation | Quality and uptime vary |
| More reliable | Lambda and similar providers | Better SLAs and enterprise reliability | Serious production workloads | More expensive |

## Recommended Starting GPU

*Prices fluctuate and vary by region/provider.*

1. **RTX 4090 / RTX 5090 (24-32 GB)**
   - Cheapest practical option
   - Roughly **$0.30-$0.50/hour** on RunPod Community or Vast.ai

2. **L40S / A6000 / RTX PRO 6000 (48 GB)**
   - Better capacity and reliability
   - Good next step if budget allows

3. **A100 40 GB / 80 GB**
   - Strong production choice
   - Higher cost

### Avoid

- GPUs with **less than 24 GB VRAM** if you want to host useful modern models.

### Recommendation

Start with **one 24 GB to 48 GB GPU**. You can always shut down the instance when not testing.

---

# Which Model to Run First

Current high-volume open models on OpenRouter are largely Chinese Flash or MoE models such as:

- DeepSeek V4.x Flash
- GLM 5.3 Flash
- MiMo-V2.6-Flash
- Qwen variants
- Nemotron
- Kimi

## Realistic Single-GPU Starter Models

### Best First Attempt

- **Qwen3**
- **Qwen2.5 14B-32B**
- AWQ, GPTQ, or FP8 quantized variants

### Other Options

- Quantized MoE models that fit into 24-48 GB VRAM
- Strong 7B-14B models if you want higher concurrency

### Avoid

- 70B+ dense models in full precision on a single consumer GPU

### Rule of Thumb

Choose a model that:

1. Already receives meaningful traffic on OpenRouter.
2. Fits comfortably on your GPU.
3. Can be served at competitive pricing and latency.

Useful resources:

- OpenRouter Rankings: https://openrouter.ai/rankings
- Individual model pages for provider and pricing information.

---

# Quick Start

## 1. Rent a GPU

Create an account on:

- RunPod
- Vast.ai

Rent a pod with one of the recommended GPUs.

Use:

- Official vLLM template, or
- Official SGLang template, or
- Ubuntu + CUDA image

---

## 2. Install and Run the Model Server

Example using **vLLM**:

```bash
pip install vllm

vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --api-key YOUR_SECRET_KEY
```

Adjust:

- Model name
- Quantization method
- Context length
- GPU utilization

Test thoroughly before exposing the endpoint publicly.

---

## 3. Expose the Endpoint

RunPod and Vast.ai typically provide:

- Public URLs
- Port forwarding
- Proxy options

Recommended setup:

```text
Internet
   ↓
Nginx/Caddy
   ↓
HTTPS
   ↓
Authentication
   ↓
vLLM or SGLang Server
```

Requirements:

- HTTPS
- API key authentication
- Streaming support
- Accurate token usage reporting

---

## 4. Implement the `/models` Endpoint

Your API should expose model metadata that OpenRouter expects, including:

- Model ID
- Context length
- Pricing
- Supported modalities
- Datacenter region
- Other provider metadata

Refer to OpenRouter provider documentation for the exact schema.

---

## 5. Prepare the Business Side

Before applying, make sure you have:

### Privacy & Compliance

- Public privacy policy
- Data retention policy
- Clear handling of user prompts and logs

### Billing

- Ability to receive payments
- Automated invoicing if possible

### Reliability

Aim for high uptime.

OpenRouter tracks:

- Availability
- Latency
- Reliability

Providers below acceptable uptime thresholds may receive little or no traffic.

---

## 6. Apply

Provider application:

https://openrouter.ai/providers/apply

Expect:

- Application backlog
- Manual review
- Possible rejection if you only have a single commodity GPU

---

# Realistic Expectations

## Pricing

You generally need one of:

- Lower prices than competitors, or
- Better latency, or
- Better regional availability

to attract traffic.

---

## Important Metrics

Monitor carefully:

- TTFT (Time To First Token)
- Tokens per second
- Error rate
- Availability
- Queue latency

These metrics strongly influence routing decisions.

---

## Use It Yourself First

A practical strategy is:

- Route your personal projects through the endpoint.
- Validate stability.
- Gather usage data.
- Improve performance before seeking external traffic.

---

## Control Costs

GPU rentals accumulate costs quickly.

Good practice:

- Stop pods when idle.
- Avoid always-on instances during experimentation.
- Track revenue versus rental expense closely.

---

## Profitability Reality

A single consumer GPU is usually **not a money-printing machine**.

Success generally requires:

- Competitive pricing
- Reliable uptime
- Good TTFT
- Consistent throughput
- Scale beyond a single GPU

Think of the first GPU as a learning platform and proof of concept.

---

# Recommended Starter Setup

| Component | Recommendation |
|------------|---------------|
| Cloud | RunPod Community Cloud |
| GPU | RTX 4090 / RTX 5090 |
| Engine | vLLM |
| Model | Qwen2.5 14B-32B AWQ |
| Reverse Proxy | Caddy or Nginx |
| HTTPS | Let's Encrypt |
| Goal | Learning + personal usage |
| Expansion Trigger | Consistent external traffic |

**Bottom line:** Start with a single RTX 4090/5090 on RunPod, serve a quantized Qwen model using vLLM, use it for your own projects first, and treat OpenRouter approval and external traffic as a bonus rather than the primary goal.
