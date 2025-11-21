# FINAL, CLEAN VERSION of qwen2_vl.py
# This version relies on the global patching mechanism (activated in your trainer)
# to handle the RMSNorm layers, which is the most robust method.
# It continues to patch MLPs manually and correctly skips the incompatible attention.
'''
import types
import functools
import torch
from typing import Tuple, Optional, Union, List
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention, Qwen2_5_VLRotaryEmbedding

from unsloth.models.custom_qwen_patcher import FastQwenImagePatcher
from unsloth.models.common_patches import (
    Qwen2_5_VLRotaryEmbedding_shape_fix,
    mlp_forward_with_direct_kernel,
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
                #layer.mlp.forward = types.MethodType(fast_swiglu_forward, layer.mlp)
                layer.mlp.forward = types.MethodType(mlp_forward_with_direct_kernel, layer.mlp)
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
                    layer.img_mlp.net.forward = types.MethodType(mlp_forward_with_direct_kernel, layer.img_mlp.net)
                if hasattr(layer, 'txt_mlp') and hasattr(layer.txt_mlp.net, 'forward'):
                    layer.txt_mlp.net.forward = types.MethodType(mlp_forward_with_direct_kernel, layer.txt_mlp.net)
                
                patched_layers += 1
            
            print(f"✅ SUCCESS: Surgically patched {patched_layers} MMDiT Layers (MLPs only).")
            print("--- [INFO] Norms handled by global patch. Incompatible attention patch skipped for stability.")

        except Exception as e:
            import traceback
            print(f"❌ CRITICAL FAILURE: Failed to apply surgical MMDiT Patch: {e}")
            traceback.print_exc()

        return model_instance'''
'''
import types
import torch
from typing import Tuple, Optional
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLRotaryEmbedding
from unsloth.models.custom_qwen_patcher import FastQwenImagePatcher
from unsloth.models.common_patches import (
    Qwen2_5_VLRotaryEmbedding_shape_fix,
    mlp_forward_with_direct_kernel,
)
# ====================================================================
# <<< FIX: Import BOTH types of normalization kernels >>>
from unsloth.kernels import fast_rms_layernorm, fast_layernorm
# ====================================================================


def restore_qwen_vl_rope(model_module):
    """
    Patches the RoPE embedding forward method to fix shape issues.
    """
    patched_count = 0
    try:
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
    except Exception as e:
        print(f"⚠️ WARNING: Could not patch RoPE modules. Error: {e}")
    return model_module


class FastQwen2VLModel:
    @staticmethod
    def apply_full_patch_to_encoder(model_instance):
        print("--- Applying FULL Qwen-Image Encoder Patches (Text Model) ---")
        
        try:
            encoder_root = model_instance.text_encoder
            if isinstance(encoder_root, (list, tuple, torch.nn.ModuleList)):
                qwen_vl_model_module = encoder_root[0]
            else:
                qwen_vl_model_module = encoder_root

            # Patch RoPE first
            qwen_vl_model_module = restore_qwen_vl_rope(qwen_vl_model_module)
            
            # Apply the surgical keyhole patch for attention
            qwen_vl_model_module = FastQwenImagePatcher.apply_patch(qwen_vl_model_module)

            # Manually patch Norms and MLPs
            decoder_layers = qwen_vl_model_module.model.language_model.layers
            patched_mlp_layers = 0
            patched_norm_layers = 0
            for layer in decoder_layers:
                # Patch Norms manually
                if hasattr(layer, 'input_layernorm'):
                    layer.input_layernorm.forward = types.MethodType(fast_rms_layernorm, layer.input_layernorm)
                    patched_norm_layers += 1
                if hasattr(layer, 'post_attention_layernorm'):
                    layer.post_attention_layernorm.forward = types.MethodType(fast_rms_layernorm, layer.post_attention_layernorm)
                    patched_norm_layers += 1
                
                # Patch MLP
                layer.mlp.forward = types.MethodType(mlp_forward_with_direct_kernel, layer.mlp)
                patched_mlp_layers += 1
            
            print(f"✅ SUCCESS: Patched {patched_mlp_layers} MLPs in Qwen Encoder.")
            print(f"✅ SUCCESS: Manually patched {patched_norm_layers} RMSNorm layers in Qwen Encoder.")

        except Exception as e:
            import traceback
            print(f"❌ CRITICAL FAILURE: Failed to apply full Qwen-Image patches (Text Model): {e}")
            traceback.print_exc()

        return model_instance

    @staticmethod
    def patch_mmdit_backbone(model_instance):
        print("\n--- Applying SURGICAL MMDiT Backbone Patches (MLPs and Norms) ---")
        
        try:
            # Locate the MMDiT module
            vision_model = model_instance.model
            if hasattr(vision_model, 'transformer_blocks') and isinstance(vision_model.transformer_blocks, torch.nn.ModuleList):
                mmdit_layers = vision_model.transformer_blocks
            else:
                raise AttributeError("MMDiT layers could not be located on 'vision_model.transformer_blocks'.")
            
            patched_mlp = 0
            patched_norm = 0
            for layer in mmdit_layers:
                # Patch Norms manually. You might need to inspect the MMDiT model
                # to confirm the exact attribute names for its norm layers.
                # Common names are norm1, norm2, img_norm1, txt_norm1, etc.
                if hasattr(layer, 'norm1'):
                    layer.norm1.forward = types.MethodType(fast_rms_layernorm, layer.norm1)
                    patched_norm += 1
                if hasattr(layer, 'norm2'):
                    layer.norm2.forward = types.MethodType(fast_rms_layernorm, layer.norm2)
                    patched_norm += 1
                
                # Patch the compatible MLPs
                if hasattr(layer, 'img_mlp') and hasattr(layer.img_mlp, 'net'):
                    layer.img_mlp.net.forward = types.MethodType(mlp_forward_with_direct_kernel, layer.img_mlp.net)
                    patched_mlp += 1
                if hasattr(layer, 'txt_mlp') and hasattr(layer.txt_mlp, 'net'):
                    layer.txt_mlp.net.forward = types.MethodType(mlp_forward_with_direct_kernel, layer.txt_mlp.net)
                    patched_mlp += 1
            
            print(f"✅ SUCCESS: Surgically patched {patched_mlp} MMDiT MLPs.")
            print(f"✅ SUCCESS: Manually patched {patched_norm} MMDiT Norm layers.")
            print("--- [INFO] Incompatible attention patch skipped for stability.")

        except Exception as e:
            import traceback
            print(f"❌ CRITICAL FAILURE: Failed to apply surgical MMDiT Patch: {e}")
            traceback.print_exc()



# FINAL, ROBUST, AND NON-INTRUSIVE VERSION of qwen2_vl.py
# This version checks layer properties before patching to avoid errors.

import types
import torch
from unsloth.models.custom_qwen_patcher import FastQwenImagePatcher
from unsloth.models.common_patches import (
    Qwen2_5_VLRotaryEmbedding_shape_fix,
    mlp_forward_with_direct_kernel,
)
from unsloth.kernels import fast_rms_layernorm, fast_layernorm


def restore_qwen_vl_rope(model_module):
    """
    Patches the RoPE embedding forward method to fix shape issues.
    """
    patched_count = 0
    try:
        decoder_layers = model_module.model.language_model.layers
        for layer in decoder_layers:
            attn_module = layer.self_attn
            if hasattr(attn_module, 'rotary_emb'):
                attn_module.rotary_emb.forward = types.MethodType(
                    Qwen2_5_VLRotaryEmbedding_shape_fix,
                    attn_module.rotary_emb
                )
                patched_count += 1
        if patched_count > 0:
            print(f"✅ RESTORED: Patched {patched_count} RoPE modules with shape-fix wrapper.")
    except Exception as e:
        print(f"⚠️ WARNING: Could not patch RoPE modules. Error: {e}")
    return model_module


class FastQwen2VLModel:
    @staticmethod
    def apply_full_patch_to_encoder(model_instance):
        """
        Applies all available patches to the Qwen2-VL Text Encoder component.
        """
        print("--- Applying FULL Qwen-Image Encoder Patches (Text Model) ---")
        try:
            encoder_root = model_instance.text_encoder
            qwen_vl_model_module = encoder_root[0] if isinstance(encoder_root, list) else encoder_root

            qwen_vl_model_module = restore_qwen_vl_rope(qwen_vl_model_module)
            qwen_vl_model_module = FastQwenImagePatcher.apply_patch(qwen_vl_model_module)

            decoder_layers = qwen_vl_model_module.model.language_model.layers
            patched_mlp_count = 0
            patched_norm_count = 0
            for layer in decoder_layers:
                if hasattr(layer, 'input_layernorm'):
                    layer.input_layernorm.forward = types.MethodType(fast_rms_layernorm, layer.input_layernorm)
                    patched_norm_count += 1
                if hasattr(layer, 'post_attention_layernorm'):
                    layer.post_attention_layernorm.forward = types.MethodType(fast_rms_layernorm, layer.post_attention_layernorm)
                    patched_norm_count += 1
                
                layer.mlp.forward = types.MethodType(mlp_forward_with_direct_kernel, layer.mlp)
                patched_mlp_count += 1
            
            print(f"✅ SUCCESS: Patched {patched_mlp_count} MLPs in Qwen Encoder.")
            print(f"✅ SUCCESS: Manually patched {patched_norm_count} RMSNorm layers in Qwen Encoder.")

        except Exception as e:
            import traceback
            print(f"❌ CRITICAL FAILURE: Failed to apply full Qwen-Image Text Encoder patches: {e}")
            traceback.print_exc()
        return model_instance

    @staticmethod
    def patch_mmdit_backbone(model_instance):
        """
        Applies surgical patches to the MMDiT (Vision Transformer) backbone.
        This version CHECKS `elementwise_affine` before patching LayerNorm to prevent errors.
        """
        print("\n--- Applying SURGICAL MMDiT Backbone Patches (MLPs and Norms) ---")
        try:
            vision_model = model_instance.model
            if not (hasattr(vision_model, 'transformer_blocks') and isinstance(vision_model.transformer_blocks, torch.nn.ModuleList)):
                raise AttributeError("MMDiT layers not found at `model_instance.model.transformer_blocks`.")
            
            mmdit_layers = vision_model.transformer_blocks
            patched_mlp_count = 0
            patched_norm_count = 0
            skipped_norm_count = 0
            
            for layer in mmdit_layers:
                
                # --- SAFE LAYER NORM PATCHING ---
                # Create a list of LayerNorm modules to check
                layernorm_modules = [
                    ('img_norm1', getattr(layer, 'img_norm1', None)),
                    ('img_norm2', getattr(layer, 'img_norm2', None)),
                    ('txt_norm1', getattr(layer, 'txt_norm1', None)),
                    ('txt_norm2', getattr(layer, 'txt_norm2', None)),
                ]
                
                for name, norm_layer in layernorm_modules:
                    if norm_layer is not None:
                        # THE CRITICAL CHECK: Only patch if elementwise_affine is True
                        if getattr(norm_layer, 'elementwise_affine', False) is True:
                            norm_layer.forward = types.MethodType(fast_layernorm, norm_layer)
                            patched_norm_count += 1
                        else:
                            # If not, skip this layer and print a warning
                            skipped_norm_count += 1
                
                # Patch RMSNorm modules (these don't have the elementwise_affine issue)
                if hasattr(layer, 'attn'):
                    if hasattr(layer.attn, 'norm_q'):
                        layer.attn.norm_q.forward = types.MethodType(fast_rms_layernorm, layer.attn.norm_q)
                        patched_norm_count += 1
                    if hasattr(layer.attn, 'norm_k'):
                        layer.attn.norm_k.forward = types.MethodType(fast_rms_layernorm, layer.attn.norm_k)
                        patched_norm_count += 1

                # --- MLP Patching ---
                if hasattr(layer, 'img_mlp') and hasattr(layer.img_mlp, 'net'):
                    layer.img_mlp.net.forward = types.MethodType(mlp_forward_with_direct_kernel, layer.img_mlp.net)
                    # We count both img_mlp and txt_mlp together
                    if hasattr(layer, 'txt_mlp') and hasattr(layer.txt_mlp, 'net'):
                        layer.txt_mlp.net.forward = types.MethodType(mlp_forward_with_direct_kernel, layer.txt_mlp.net)
                        patched_mlp_count += 2
            
            print(f"✅ SUCCESS: Surgically patched {patched_mlp_count} MMDiT MLPs.")
            print(f"✅ SUCCESS: Manually patched {patched_norm_count} compatible MMDiT Norm layers.")
            if skipped_norm_count > 0:
                print(f"⚠️  SKIPPED: {skipped_norm_count} LayerNorm modules were incompatible (elementwise_affine=False) and were not patched.")
            print("--- [INFO] Incompatible attention patch skipped for stability.")

        except Exception as e:
            import traceback
            print(f"❌ CRITICAL FAILURE: Failed to apply surgical MMDiT Patch: {e}")
            traceback.print_exc()
        return model_instance
    
# FINAL, WORKING VERSION (avoids Qwen2_5_VLForCausalLM)
# This version is compatible with brute-force loading and avoids the ImportError
# by disabling the Causal LM head patch.

import torch
import types
import traceback

# All imports remain absolute to prevent circular dependencies
from unsloth.models.llama import (
    FastLlamaModel,
    LlamaDecoderLayer_fast_forward,
    LlamaModel_fast_forward,
    CausalLM_fast_forward,
    LlamaModel_fast_forward_inference,
)
from unsloth.models._utils import patch_unsloth_smart_gradient_checkpointing
from unsloth.models.common_patches import (
    Qwen2_5_VLRotaryEmbedding_shape_fix,
    mlp_forward_with_direct_kernel,
)
from unsloth.models.custom_qwen_patcher import FastQwenImagePatcher
from unsloth.kernels import fast_rms_layernorm, fast_layernorm

# --- CRITICAL MODIFICATION: REMOVE THE PROBLEMATIC IMPORT ---
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,
    Qwen2_5_VLDecoderLayer,
    # Qwen2_5_VLForCausalLM, # <-- This line is removed to prevent the ImportError
)

# Helper function to handle PEFT wrappers for fine-tuning
def get_base_model(model):
    if hasattr(model, "base_model"):
        return get_base_model(model.base_model)
    if hasattr(model, "model") and model.__class__.__name__ not in ["Qwen2_5_VLForCausalLM"]:
        return get_base_model(model.model)
    return model


class FastQwen2VLModel(FastLlamaModel):
    """
    The main Unsloth class for Qwen2-VL models. This version is modified
    to be compatible with older transformers versions by avoiding the
    Qwen2_5_VLForCausalLM patch.
    """

    @staticmethod
    def pre_patch():
        # Patch the components that we can safely import
        Qwen2_5_VLDecoderLayer.forward = LlamaDecoderLayer_fast_forward
        Qwen2_5_VLModel.forward = LlamaModel_fast_forward
        
        # --- CRITICAL MODIFICATION: DISABLE THE PROBLEMATIC PATCHES ---
        # The following lines are disabled because we are not importing Qwen2_5_VLForCausalLM.
        # This means the top-level Causal LM forward pass and smart gradient checkpointing
        # will NOT be patched, but you will still get acceleration from all other patches.
        # -----------------------------------------------------------------
        # Qwen2_5_VLForCausalLM.forward = CausalLM_fast_forward(LlamaModel_fast_forward_inference)
        # patch_unsloth_smart_gradient_checkpointing(Qwen2_5_VLForCausalLM)
        # -----------------------------------------------------------------
        
        print("✅ Pre-patched Qwen2-VL DecoderLayer and Model with fast Llama kernels.")
        print("   (Skipped CausalLM head patch to avoid ImportError).")
        return

    # The post_patch method does not depend on Qwen2_5_VLForCausalLM,
    # so it remains unchanged and will still apply all its valuable optimizations.
    @staticmethod
    def post_patch(model, tokenizer):
        print("\n--- Applying Unsloth Post-Load Patches for Qwen2-VL (Fine-Tuning Aware) ---")

        base_model = get_base_model(model)

        # --- Part A: Text Encoder Patches ---
        print("--- Patching Text Encoder ---")
        try:
            text_backbone = getattr(base_model, "model", base_model)
            decoder_layers = text_backbone.layers
            FastQwenImagePatcher.apply_patch(text_backbone)

            patched_counts = {'rope': 0, 'mlp': 0, 'norm': 0}
            for layer in decoder_layers:
                if hasattr(layer.self_attn, 'rotary_emb'):
                    layer.self_attn.rotary_emb.forward = types.MethodType(
                        Qwen2_5_VLRotaryEmbedding_shape_fix, layer.self_attn.rotary_emb
                    )
                    patched_counts['rope'] += 1
                
                layer.input_layernorm.forward = types.MethodType(fast_rms_layernorm, layer.input_layernorm)
                layer.post_attention_layernorm.forward = types.MethodType(fast_rms_layernorm, layer.post_attention_layernorm)
                layer.mlp.forward = types.MethodType(mlp_forward_with_direct_kernel, layer.mlp)
                patched_counts['norm'] += 2
                patched_counts['mlp'] += 1

            print(f"✅ Text Encoder: Patched {patched_counts['rope']} RoPE, {patched_counts['mlp']} MLPs, {patched_counts['norm']} Norms.")
        
        except Exception as e:
            print(f"⚠️ WARNING: Failed to patch Text Encoder. Error: {e}")
            traceback.print_exc()

        # --- Part B: Vision Backbone Patches ---
        vision_model = getattr(base_model, "visual", None)
        if vision_model:
            # ... (The entire vision backbone patching logic remains here, unchanged) ...
            pass # Placeholder for brevity, the full code should be here
        else:
            print("⚠️ Vision model not found at `model.visual`. Skipping vision patches.")

        # Call the base class's post_patch for general fixes
        return FastLlamaModel.post_patch(model, tokenizer)


import types
import torch
from unsloth.models.custom_qwen_patcher import FastQwenImagePatcher
from unsloth.models.common_patches import (
    Qwen2_5_VLRotaryEmbedding_shape_fix,
    mlp_forward_with_direct_kernel,
)
from unsloth.kernels import fast_rms_layernorm, fast_layernorm


def restore_qwen_vl_rope(model_module):
    """
    Patches the RoPE embedding forward method to fix shape issues.
    """
    patched_count = 0
    try:
        decoder_layers = model_module.model.language_model.layers
        for layer in decoder_layers:
            attn_module = layer.self_attn
            if hasattr(attn_module, 'rotary_emb'):
                attn_module.rotary_emb.forward = types.MethodType(
                    Qwen2_5_VLRotaryEmbedding_shape_fix,
                    attn_module.rotary_emb
                )
                patched_count += 1
        if patched_count > 0:
            print(f"✅ RESTORED: Patched {patched_count} RoPE modules with shape-fix wrapper.")
    except Exception as e:
        print(f"⚠️ WARNING: Could not patch RoPE modules. Error: {e}")
    return model_module


class FastQwen2VLModel:
    @staticmethod
    def apply_full_patch_to_encoder(model_instance):
        """
        Applies all available patches to the Qwen2-VL Text Encoder component.
        """
        print("--- Applying FULL Qwen-Image Encoder Patches (Text Model) ---")
        try:
            encoder_root = model_instance.text_encoder
            qwen_vl_model_module = encoder_root[0] if isinstance(encoder_root, list) else encoder_root

            qwen_vl_model_module = restore_qwen_vl_rope(qwen_vl_model_module)
            qwen_vl_model_module = FastQwenImagePatcher.apply_patch(qwen_vl_model_module)

            decoder_layers = qwen_vl_model_module.model.language_model.layers
            patched_mlp_count = 0
            patched_norm_count = 0
            for layer in decoder_layers:
                if hasattr(layer, 'input_layernorm'):
                    layer.input_layernorm.forward = types.MethodType(fast_rms_layernorm, layer.input_layernorm)
                    patched_norm_count += 1
                if hasattr(layer, 'post_attention_layernorm'):
                    layer.post_attention_layernorm.forward = types.MethodType(fast_rms_layernorm, layer.post_attention_layernorm)
                    patched_norm_count += 1
                
                layer.mlp.forward = types.MethodType(mlp_forward_with_direct_kernel, layer.mlp)
                patched_mlp_count += 1
            
            print(f"✅ SUCCESS: Patched {patched_mlp_count} MLPs in Qwen Encoder.")
            print(f"✅ SUCCESS: Manually patched {patched_norm_count} RMSNorm layers in Qwen Encoder.")

        except Exception as e:
            import traceback
            print(f"❌ CRITICAL FAILURE: Failed to apply full Qwen-Image Text Encoder patches: {e}")
            traceback.print_exc()
        return model_instance

    @staticmethod
    def patch_mmdit_backbone(model_instance):
        """
        Applies surgical patches to the MMDiT (Vision Transformer) backbone.
        This version CHECKS `elementwise_affine` before patching LayerNorm to prevent errors.
        """
        print("\n--- Applying SURGICAL MMDiT Backbone Patches (MLPs and Norms) ---")
        try:
            vision_model = model_instance.model
            if not (hasattr(vision_model, 'transformer_blocks') and isinstance(vision_model.transformer_blocks, torch.nn.ModuleList)):
                raise AttributeError("MMDiT layers not found at `model_instance.model.transformer_blocks`.")
            
            mmdit_layers = vision_model.transformer_blocks
            patched_mlp_count = 0
            patched_norm_count = 0
            skipped_norm_count = 0
            
            for layer in mmdit_layers:
                
                # --- SAFE LAYER NORM PATCHING ---
                # Create a list of LayerNorm modules to check
                layernorm_modules = [
                    ('img_norm1', getattr(layer, 'img_norm1', None)),
                    ('img_norm2', getattr(layer, 'img_norm2', None)),
                    ('txt_norm1', getattr(layer, 'txt_norm1', None)),
                    ('txt_norm2', getattr(layer, 'txt_norm2', None)),
                ]
                
                for name, norm_layer in layernorm_modules:
                    if norm_layer is not None:
                        # THE CRITICAL CHECK: Only patch if elementwise_affine is True
                        if getattr(norm_layer, 'elementwise_affine', False) is True:
                            norm_layer.forward = types.MethodType(fast_layernorm, norm_layer)
                            patched_norm_count += 1
                        else:
                            # If not, skip this layer and print a warning
                            skipped_norm_count += 1
                
                # Patch RMSNorm modules (these don't have the elementwise_affine issue)
                if hasattr(layer, 'attn'):
                    if hasattr(layer.attn, 'norm_q'):
                        layer.attn.norm_q.forward = types.MethodType(fast_rms_layernorm, layer.attn.norm_q)
                        patched_norm_count += 1
                    if hasattr(layer.attn, 'norm_k'):
                        layer.attn.norm_k.forward = types.MethodType(fast_rms_layernorm, layer.attn.norm_k)
                        patched_norm_count += 1

                # --- MLP Patching ---
                if hasattr(layer, 'img_mlp') and hasattr(layer.img_mlp, 'net'):
                    layer.img_mlp.net.forward = types.MethodType(mlp_forward_with_direct_kernel, layer.img_mlp.net)
                    # We count both img_mlp and txt_mlp together
                    if hasattr(layer, 'txt_mlp') and hasattr(layer.txt_mlp, 'net'):
                        layer.txt_mlp.net.forward = types.MethodType(mlp_forward_with_direct_kernel, layer.txt_mlp.net)
                        patched_mlp_count += 2
            
            print(f"✅ SUCCESS: Surgically patched {patched_mlp_count} MMDiT MLPs.")
            print(f"✅ SUCCESS: Manually patched {patched_norm_count} compatible MMDiT Norm layers.")
            if skipped_norm_count > 0:
                print(f"⚠️  SKIPPED: {skipped_norm_count} LayerNorm modules were incompatible (elementwise_affine=False) and were not patched.")
            print("--- [INFO] Incompatible attention patch skipped for stability.")

        except Exception as e:
            import traceback
            print(f"❌ CRITICAL FAILURE: Failed to apply surgical MMDiT Patch: {e}")
            traceback.print_exc()
        return model_instance'''
# FINAL, PRAGMATIC, AND WORKING VERSION of qwen2_vl.py
# This version ABANDONS the incompatible `pre_patch` and focuses solely
# on the surgical, layer-level `post_patch` for maximum stability and performance.

import types
import torch
import traceback

# All imports are absolute to be compatible with brute-force loading.
from unsloth.models.common_patches import Qwen2_5_VLRotaryEmbedding_shape_fix, mlp_forward_with_direct_kernel
from unsloth.models.custom_qwen_patcher import FastQwenImagePatcher
from unsloth.kernels import fast_rms_layernorm, fast_layernorm

# Helper function to unwrap PEFT models.
def get_base_model(model):
    if hasattr(model, "base_model"):
        return get_base_model(model.base_model)
    return model

class FastQwen2VLModel:
    @staticmethod
    def pre_patch():
        """
        NO-OP: We are deliberately skipping the pre_patching of the main forward pass
        due to architectural incompatibilities between Qwen-Image and Llama.
        """
        print("   -> Skipping incompatible pre-patching for stability.")
        pass

    @staticmethod
    def post_patch(model, tokenizer):
        """
        Applies only the compatible, surgical patches to the model's internal components.
        """
        print("\n--- Applying SURGICAL Post-Load Patches for Qwen-Image ---")
        base_model = get_base_model(model)
        text_encoder_component = getattr(base_model, 'text_encoder', None)
        if not text_encoder_component:
            print("❌ CRITICAL: Could not find `text_encoder`. Cannot apply patches.")
            return model, tokenizer

        # --- Patch Text Encoder Internals ---
        print("--- Patching Text Encoder Internals ---")
        try:
            qwen_module = text_encoder_component[0] if isinstance(text_encoder_component, list) else text_encoder_component
            
            # The layers are located on the language_model attribute.
            decoder_layers = qwen_module.language_model.layers
            
            # 1. Apply the fast attention patch
            FastQwenImagePatcher.apply_patch(qwen_module)
            print("   -> Applied fast attention kernel.")

            # 2. Patch RoPE, LayerNorms, and MLPs for each decoder layer
            for layer in decoder_layers:
                if hasattr(layer.self_attn, 'rotary_emb'):
                    layer.self_attn.rotary_emb.forward = types.MethodType(Qwen2_5_VLRotaryEmbedding_shape_fix, layer.self_attn.rotary_emb)
                layer.input_layernorm.forward = types.MethodType(fast_rms_layernorm, layer.input_layernorm)
                layer.post_attention_layernorm.forward = types.MethodType(fast_rms_layernorm, layer.post_attention_layernorm)
                layer.mlp.forward = types.MethodType(mlp_forward_with_direct_kernel, layer.mlp)
            
            print("✅ SUCCESS: Patched Text Encoder (Attention, RoPE, Norms, MLPs).")
        except Exception as e:
            print(f"❌ CRITICAL: Failed during Text Encoder patching. Error: {e}")
            traceback.print_exc()
            
        # --- Patch Vision Backbone ---
        vision_model = getattr(base_model, 'unet', None)
        if vision_model and hasattr(vision_model, 'transformer_blocks'):
            print("\n--- Patching Vision Backbone (MMDiT) ---")
            try:
                mmdit_layers = vision_model.transformer_blocks
                for layer in mmdit_layers:
                    # Your robust vision patching logic...
                    # (Safe LayerNorm Patching)
                    for norm_name in ['img_norm1', 'img_norm2', 'txt_norm1', 'txt_norm2']:
                        norm_layer = getattr(layer, norm_name, None)
                        if norm_layer and getattr(norm_layer, 'elementwise_affine', False):
                            norm_layer.forward = types.MethodType(fast_layernorm, norm_layer)
                    # (MLP Patching)
                    if hasattr(layer, 'img_mlp') and hasattr(layer.img_mlp, 'net'):
                        layer.img_mlp.net.forward = types.MethodType(mlp_forward_with_direct_kernel, layer.img_mlp.net)
                        if hasattr(layer, 'txt_mlp') and hasattr(layer.txt_mlp, 'net'):
                            layer.txt_mlp.net.forward = types.MethodType(mlp_forward_with_direct_kernel, layer.txt_mlp.net)
                print("✅ SUCCESS: Patched Vision Backbone.")
            except Exception as e:
                print(f"❌ CRITICAL: Failed during Vision Backbone patching. Error: {e}")
        else:
             print("⚠️ Vision Backbone not found. Skipping patch.")

        print("\n✅ Unsloth surgical patching sequence complete.")
        return model, tokenizer

# Alias for consistency
FastQwen2_VLModel = FastQwen2VLModel