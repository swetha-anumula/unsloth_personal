# In pure_hf_test.py
from transformers import AutoConfig

model_name = "Qwen/Qwen-Image"
print(f"Attempting to download config for '{model_name}' using pure transformers...")

try:
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    print("\n✅ SUCCESS: Hugging Face transformers library successfully downloaded the config.")
    print("This means your login and network connection are working correctly.")
    print("\nModel Type:", config.model_type)

except Exception as e:
    print(f"\n❌ FAILED: Pure transformers also failed to download the config.")
    print(f"   Error: {e}")
    print("\n   This confirms the issue is with your environment, not your Unsloth modifications.")
    print("   Possible causes: Firewall, proxy server, or an issue with the saved token.")
