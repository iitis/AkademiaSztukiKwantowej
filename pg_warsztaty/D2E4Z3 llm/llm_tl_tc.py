import os
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig
from datasets import load_dataset

# ------------------------------
# CONFIGURATION
# ------------------------------

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_dir = "adapter_model"  # Folder to save/load LoRA
use_fp16 = True  # Use float16 for speed (GPU only)

# ------------------------------
# LOAD BASE MODEL AND TOKENIZER
# ------------------------------

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # Required to avoid pad token errors

base_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
base_model = prepare_model_for_kbit_training(base_model)
base_model.gradient_checkpointing_disable()  # Avoids runtime errors in causal masking

# ------------------------------
# DATA PREPARATION
# ------------------------------

print("Loading small sample of SST2 (sentiment dataset)...")
ds = load_dataset("sst2", split="train[:200]")  # 200 examples: good enough for demo
label_map = {0: " negative", 1: " positive"}

def format_prompt(example):
    """Create a prompt and tokenized input for each example"""
    prompt = f"### Review:\n{example['sentence']}\n### Sentiment:"
    full = prompt + label_map[example["label"]]
    encoded = tokenizer(full, padding='max_length', truncation=True, max_length=128)
    encoded["labels"] = encoded["input_ids"].copy()
    return encoded

ds = ds.map(format_prompt, remove_columns=ds.column_names)

# ------------------------------
# SHOW EXAMPLES
# ------------------------------

print("\nSample formatted prompts:")
for i in range(3):
    ex = ds[i]
    print(tokenizer.decode(ex["input_ids"], skip_special_tokens=True))

# ------------------------------
# CHECK FOR SAVED ADAPTER
# ------------------------------

if os.path.exists(adapter_dir):
    print(f"\n🔁 Found existing LoRA adapter in {adapter_dir}, loading...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
else:
    print("\n🛠️  No adapter found. Training LoRA from scratch...")

    # Apply LoRA to base model
    model = get_peft_model(base_model, LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM", bias="none"
    ))

    # Training
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        fp16=use_fp16,
        logging_steps=10,
        report_to="none",
        save_strategy="no"
    )

    trainer = Trainer(model=model, train_dataset=ds, args=args)
    trainer.train()

    # Save adapter
    model.save_pretrained(adapter_dir)
    print(f"✅ LoRA adapter saved to {adapter_dir}")
    model.merge_and_unload()  # Optional: merge for inference

# ------------------------------
# SIMPLE TEXT CLASSIFICATION LOOP
# ------------------------------

print("\n🤖 Ready! Ask about any review. Type 'exit' to quit.")
while True:
    user_input = input("\nYour review: ").strip()
    if user_input.lower() in {"exit", "quit"}:
        break

    prompt = f"### Review:\n{user_input}\n### Sentiment:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
