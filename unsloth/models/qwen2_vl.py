# FINAL, CLEAN VERSION of qwen2_vl.py
# This version relies on the global patching mechanism (activated in your trainer)
# to handle the RMSNorm layers, which is the most robust method.
# It continues to patch MLPs manually and correctly skips the incompatible attention.

import types
import functools
import torch
from typing import Tuple, Optional, Union, List
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention, Qwen2_5_VLRotaryEmbedding

from unsloth.models.custom_qwen_patcher import FastQwenImagePatcher
from unsloth.models.common_patches import (
    Qwen2_5_VLRotaryEmbedding_shape_fix,
    fast_swiglu_forward,
    # We no longer need to import fast_rms_layernorm here, as it's handled globally.
)

def restore_qwen_vl_rope(model_module):
    patched_count = 0
    decoder_layers = model_module.model.language_model.layers
    for layer in decoder_layers:
        attn_module = layer.self_attn
        if hasattr(attn_module, 'rotary_emb'):
            attn_module.rotary_emb.forward = types.MethodType(
                Qwen2_5_VLRotaryEmbedding_shape_fix,
                attn_module.rotary_emb
            )
            patched_count += 1
    print(f"✅ RESTORED: Patched {patched_count} RoPE modules with shape-fix wrapper.")
    return model_module

class FastQwen2VLModel:
    @staticmethod
    def apply_full_patch_to_encoder(model_instance):
        print("--- Applying FULL Qwen-Image Encoder Patches (Text Model) ---")
        print('patching encoder')
        try:
            encoder_root = model_instance.text_encoder
            if isinstance(encoder_root, (list, tuple, torch.nn.ModuleList)):
                qwen_vl_model_module = encoder_root[0]
            else:
                qwen_vl_model_module = encoder_root

            decoder_layers = qwen_vl_model_module.model.language_model.layers
            qwen_vl_model_module = restore_qwen_vl_rope(qwen_vl_model_module)
            qwen_vl_model_module = FastQwenImagePatcher.apply_patch(qwen_vl_model_module)

            patched_layers = 0
            for layer in decoder_layers:
                # ====================================================================
                # <<< CLEANUP: Rely on global patch for Norms >>>
                # The manual MethodType patching for Norms is no longer needed.
                # if hasattr(layer, 'input_layernorm'):
                #     layer.input_layernorm.forward = types.MethodType(fast_rms_layernorm, layer.input_layernorm)
                # if hasattr(layer, 'post_attention_layernorm'):
                #     layer.post_attention_layernorm.forward = types.MethodType(fast_rms_layernorm, layer.post_attention_layernorm)
                # ====================================================================
                
                # We still patch the MLP manually as it's clean and effective.
                layer.mlp.forward = types.MethodType(fast_swiglu_forward, layer.mlp)
                patched_layers += 1
            print(f"✅ SUCCESS: Patched {patched_layers} MLPs in Qwen Encoder. (Norms handled globally)")

        except Exception as e:
            import traceback
            print(f"❌ CRITICAL FAILURE: Failed to apply full Qwen-Image patches (Text Model): {e}")
            traceback.print_exc()

        return model_instance

    @staticmethod
    def patch_mmdit_backbone(model_instance, run_as_inspector=False):
        print("\n--- Applying SURGICAL MMDiT Backbone Patches (MLPs only) ---")
        patched_layers = 0
        try:
            vision_model = model_instance.model
            mmdit_layers = None
            if hasattr(vision_model, 'transformer_blocks') and isinstance(vision_model.transformer_blocks, torch.nn.ModuleList):
                mmdit_layers = vision_model.transformer_blocks
            else:
                raise AttributeError("MMDiT layers could not be located on 'vision_model.transformer_blocks'.")
            
            for layer in mmdit_layers:
                # ====================================================================
                # <<< CLEANUP: Rely on global patch for Norms >>>
                # The manual MethodType patching for Norms is no longer needed.
                # if hasattr(layer, 'img_norm1'):
                #     layer.img_norm1.forward = types.MethodType(fast_rms_layernorm, layer.img_norm1)
                # ... (and for img_norm2, txt_norm1, txt_norm2) ...
                # ====================================================================

                # SKIP Attention Patch (Incompatible)

                # Patch the compatible MLPs
                if hasattr(layer, 'img_mlp') and hasattr(layer.img_mlp.net, 'forward'):
                    layer.img_mlp.net.forward = types.MethodType(fast_swiglu_forward, layer.img_mlp.net)
                if hasattr(layer, 'txt_mlp') and hasattr(layer.txt_mlp.net, 'forward'):
                    layer.txt_mlp.net.forward = types.MethodType(fast_swiglu_forward, layer.txt_mlp.net)
                
                patched_layers += 1
            
            print(f"✅ SUCCESS: Surgically patched {patched_layers} MMDiT Layers (MLPs only).")
            print("--- [INFO] Norms handled by global patch. Incompatible attention patch skipped for stability.")

        except Exception as e:
            import traceback
            print(f"❌ CRITICAL FAILURE: Failed to apply surgical MMDiT Patch: {e}")
            traceback.print_exc()

        return model_instance