# ✱ ONE-CELL DEMO: prompt TinyLlama-1.1B-Chat and print its reply ✱
#
# ▶ Requires ≈2 GB RAM if you let Transformers load it in 4-bit.
# ▶ Works on laptop GPUs or free-tier Colab (T4 / K80).
# -----------------------------------------------------------------

# 1.  Install minimal deps (≈ 2-3 min the first time)
# !pip install -q transformers accelerate bitsandbytes

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # open-weights on HF Hub

# 2.  Load tokenizer + 4-bit quantised model onto GPU/CPU automatically
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,       # bitsandbytes quantisation → tiny VRAM
    device_map="auto",       # sends weights to first CUDA device if available
    bnb_4bit_compute_dtype=torch.float16,   # <─ compute in FP16
    bnb_4bit_quant_type="nf4",              # (default is fine, just explicit)
    bnb_4bit_use_double_quant=True          # optional, tiny accuracy boost
)

# 3.  Quick generation pipeline
chat = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = (
    "### User: In one sentence, explain why the sky is blue.\n"
    "### Assistant:"
)

# prompt = (
#     "### User: Why is a raven like a writing desk?\n"
#     "### Assistant:"
# )

response = chat(
    prompt,
    max_new_tokens=1024,
    temperature=0.3,#0.7,
    do_sample=True,
)[0]["generated_text"]

print(response)