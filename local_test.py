# In local_test.py
from unsloth import FastLanguageModel

print("🧪 Starting Local Cache Test...")
print("This will force Unsloth to load the model from your disk and not use the internet.")

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen-Image",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        trust_remote_code=True,
        # --- THIS IS THE CRITICAL CHANGE ---
        local_files_only=True,
        # -----------------------------------
    )
    print("\n✅✅✅  SUCCESS! The model loaded from your local cache!")
    print("This confirms the issue is a network problem, and we have found the workaround.")
    print("You can now proceed with your training script by adding 'local_files_only=True'.")

except Exception as e:
    print(f"\n❌ FAILED: The model could not be loaded even from the local cache.")
    print(f"   Error: {e}")
    print("\n   This means the model files in your cache are incomplete or corrupted.")
    print("   The only solution is to fix the network issue so the model can be re-downloaded correctly.")
