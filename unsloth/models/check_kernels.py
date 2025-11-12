import torch
from unsloth.models.common_patches import fast_rms_layernorm
from transformers.models.llama.modeling_llama import LlamaRMSNorm

# 1. Create a standard LayerNorm module, just like in the model
device = "cuda"
dtype = torch.bfloat16
layernorm_module = LlamaRMSNorm((4096,), eps=1e-5).to(device).to(dtype)

# 2. Create a dummy input tensor
input_tensor = torch.randn(4, 2048, 4096, device=device, dtype=dtype)

# 3. Define the function we want to profile
def run_patched_layernorm():
    # This is the exact function our patch calls
    output = fast_rms_layernorm(layernorm_module, input_tensor)
    return output

# 4. Run the PyTorch Profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    with torch.profiler.record_function("unsloth_layernorm_test"):
        run_patched_layernorm()

# 5. Print the results, sorted by CUDA time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))