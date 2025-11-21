# my_llama_patcher.py
#
# A surgical patcher that applies your kernel patching philosophy to a
# standard, pre-loaded Llama model instance.

import types
import torch
import traceback

# All imports are absolute to be compatible with external loading.
from unsloth.models.common_patches import mlp_forward_with_direct_kernel
from unsloth.kernels import fast_rms_layernorm


def get_base_model(model):
    if hasattr(model, "base_model") and model.base_model is not model:
        return get_base_model(model.base_model)
    return model

def apply_my_surgical_patches(model):
    print("\n--- Applying YOUR custom surgical patches to the Llama model ---")
    try:
        base_model = get_base_model(model)
        transformer_model = getattr(base_model, 'model', base_model)
        
        print("--- Patching Llama Internals (MLPs & Norms) ---")
        
        decoder_layers = transformer_model.layers
        patched_counts = {'mlps': 0, 'norms': 0}
        for idx, layer in enumerate(decoder_layers):
            layer.input_layernorm.forward = types.MethodType(fast_rms_layernorm, layer.input_layernorm)
            layer.post_attention_layernorm.forward = types.MethodType(fast_rms_layernorm, layer.post_attention_layernorm)
            patched_counts['norms'] += 2
            '''if hasattr(layer, 'mlp'):
                layer.mlp.forward = types.MethodType(mlp_forward_with_direct_kernel, layer.mlp)
                patched_counts['mlps'] += 1'''
                
        print(f"✅ SUCCESS: Patched {patched_counts['norms']} RMSNorm layers.")
        #print(f"✅ SUCCESS: Patched {patched_counts['mlps']} MLP layers.")

    except Exception as e:
        print(f"❌ CRITICAL FAILURE during custom patching. Performance will not be improved. Error: {e}")
        traceback.print_exc()
        
    # Return the patched model.
    return model