# hf_check.py
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

model_name = "Qwen/Qwen-Image" 
try:
    # Use AutoConfig first to test name resolution
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    print(f"✅ HF Check: Config loaded successfully! Model type: {config.model_type}")

    # Now try the actual model load
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True
    )
    print("✅ HF Check: Model loaded successfully!")

except Exception as e:
    print(f"❌ HF Check FAILED: {e}")
    print("This confirms the issue is with the environment or the model name itself, not your patching code.")
