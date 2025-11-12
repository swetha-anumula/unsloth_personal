# In patch_trial.py

from diffusers import DiffusionPipeline
import torch
import sys
import os

# Set up path and imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))) 
from unsloth.models.custom_qwen_patcher import FastQwenImagePatcher
# You might need to import Qwen2_5_VLAttention for isinstance check 
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention 

model_name = "Qwen/Qwen-Image" 
EXPECTED_ATTN_FORWARD = "Qwen2_5_VLAttention_keyhole_forward"

# --- Custom Pipeline Loading and Patching ---
def load_and_patch_diffusers_pipeline(model_name):
    # 1. Load the pipeline using diffusers
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"

    print(f"--- 1. Loading Diffusers Pipeline: {model_name} ---")
    # Using 'trust_remote_code=True' is CRITICAL for Qwen models
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype, trust_remote_code=True)
    pipe = pipe.to(device)

    # 2. Apply the custom patch directly to the loaded pipe object
    pipe = FastQwenImagePatcher.apply_patch(pipe)
    
    return pipe

# --- Execution ---
try:
    pipe = load_and_patch_diffusers_pipeline(model_name)
    
    print("\n--- 2. Verifying Patch Application ---")
    
    # 3. Verify the patch (Using the CORRECT, inspected path: pipe.text_encoder)
    # The full path to the attention module inside Qwen2_5_VLForConditionalGeneration:
    first_attn_module = pipe.text_encoder.model.language_model.layers[0].self_attn
    patched_func_name = first_attn_module.forward.__name__

    print(f"\n--- VERIFICATION RESULTS ---")
    print(f"Patched function name found: {patched_func_name}")
    if patched_func_name == EXPECTED_ATTN_FORWARD:
        print("✅ SUCCESS: Diffusers Pipeline's Qwen2.5-VL Attention is patched.")
    else:
        print("❌ FAILURE: Patching failed. Check custom_qwen_patcher.py logic.")
        
    # Optional: Run dummy inference (Requires more setup, so skip for now)
    
except Exception as e:
    # Print the error only if it's not the simple 'pipe not defined' error
    print(f"❌ CRITICAL FAILURE: Pipeline patching/verification failed.")
    print(f"Error: {e}")