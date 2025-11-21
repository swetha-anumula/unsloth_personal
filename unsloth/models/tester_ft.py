from unsloth import FastLanguageModel
import torch
import time
import gc
import importlib.util
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer
model_name = "meta-llama/Llama-3.2-1B"
patcher_script_path = "./my_llama_patcher.py" 
dataset_name = "databricks/databricks-dolly-15k"
max_seq_length = 2048
num_train_steps = 64
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

if "YOUR_TOKEN_HERE" in HF_TOKEN:
    raise ValueError("Please open this script and replace 'hf_YOUR_TOKEN_HERE' with your actual Hugging Face token.")
print(f"INFO: Loading and preparing dataset: {dataset_name}...")
dataset = load_dataset(dataset_name, split="train")
def format_prompt(ex): return {"text": f"Instruction:\n{ex['instruction']}\n\nResponse:\n{ex['response']}"}
formatted_dataset = dataset.map(format_prompt)
def run_speed_test(model, tokenizer, dataset, test_name):
    print("\n" + "="*80); print(f"🚀 STARTING SPEED TEST: {test_name}"); print("="*80)
    
    lora_config = LoraConfig(r=16, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_dropout=0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, dataset_text_field="text", max_seq_length=max_seq_length, packing=False, args=TrainingArguments(per_device_train_batch_size=4, gradient_accumulation_steps=2, warmup_steps=10, max_steps=num_train_steps, learning_rate=2e-4, fp16=(dtype == torch.float16), bf16=(dtype == torch.bfloat16), logging_steps=1, optim="adamw_8bit", output_dir="outputs", report_to="none"))
    
    print(f"INFO: Starting training for {num_train_steps} steps...")
    start_time = time.time(); trainer.train(); end_time = time.time()
    total_time = end_time - start_time; steps_per_second = num_train_steps / total_time
    
    print(f"\n✅ FINISHED SPEED TEST: {test_name}"); print(f"   -> Measured Speed: {steps_per_second:.2f} steps/second"); print("="*80)
    
    return steps_per_second
print(f"\n--- Loading BASELINE model ({model_name} in {dtype}) ---")
baseline_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto", token=HF_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

baseline_speed = run_speed_test(baseline_model, tokenizer, formatted_dataset, "Hugging Face Baseline")
del baseline_model, tokenizer; gc.collect(); torch.cuda.empty_cache()

print(f"\n--- Loading model for YOUR CUSTOM PATCHING ({model_name} in {dtype}) ---")
model_to_patch = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto", token=HF_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

try:
    print("🚀 Applying your custom patches via brute-force import...")
    spec = importlib.util.spec_from_file_location("my_llama_patcher", patcher_script_path)
    patcher_module = importlib.util.module_from_spec(spec)
    sys.modules["my_llama_patcher"] = patcher_module
    spec.loader.exec_module(patcher_module)
    
    patch_function = patcher_module.apply_my_surgical_patches
    patched_model = patch_function(model_to_patch)
    
    print(" Your custom patching sequence was applied successfully!")
    
    custom_patched_speed = run_speed_test(patched_model, tokenizer, formatted_dataset, "Your Custom Patched Version")
except Exception as e:
    import traceback
    print(f"❌ FAILED to apply custom patches. Cannot run the second test. Error: {e}")
    traceback.print_exc()
    custom_patched_speed = 0.0

del model_to_patch, patched_model, tokenizer; gc.collect(); torch.cuda.empty_cache()

print("\n\n" + "*"*80); print(f"      PERFORMANCE COMPARISON (Model: {model_name})"); print("*"*80)
print(f"Hugging Face Baseline Speed  : {baseline_speed:.2f} steps/second")
if custom_patched_speed > 0:
    print(f"Your Custom Patched Speed    : {custom_patched_speed:.2f} steps/second")
    print("-" * 80); speedup_factor = custom_patched_speed / baseline_speed
    print(f"🔥 Your Speedup Factor: {speedup_factor:.2f}x FASTER")
else: print("Your Custom Patched Speed    : FAILED")
print("*"*80)