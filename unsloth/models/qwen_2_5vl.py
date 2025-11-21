import types
import torch
import traceback
from .llama import FastLlamaModel, LlamaDecoderLayer_fast_forward, LlamaModel_fast_forward
from .common_patches import Qwen2_5_VLRotaryEmbedding_shape_fix, mlp_forward_with_direct_kernel
from .custom_qwen_patcher import FastQwenImagePatcher
from ..kernels import fast_rms_layernorm, fast_layernorm
def get_base_model(model):
    if hasattr(model, "base_model"):
        return get_base_model(model.base_model)
    return model

class FastQwen2_5VLModel(FastLlamaModel):
    @staticmethod
    def pre_patch():
        print("   -> Applying Unsloth pre-patches for Qwen2.5-VL...")
        try:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel, Qwen2_5_VLDecoderLayer
            Qwen2_5_VLDecoderLayer.forward = LlamaDecoderLayer_fast_forward
            Qwen2_5_VLModel.forward = LlamaModel_fast_forward
            print("✅ Pre-patched Qwen2.5-VL DecoderLayer and Model.")
        except Exception as e:
            print(f"❌ ERROR during Qwen2.5-VL pre-patching: {e}")

    @staticmethod
    def post_patch(model, tokenizer):
        print("\n--- Applying Unsloth Post-Load Patches for Qwen2.5-VL ---")
        base_model = get_base_model(model)
        try:
            text_backbone_instance = getattr(base_model, "model", None)
            language_model_instance = getattr(base_model, "language_model", None)
            if not text_backbone_instance or not language_model_instance:
                raise AttributeError("Could not find '.model' or '.language_model' on the text encoder.")
            print("   -> Building compatibility bridges...")
            text_backbone_instance.embed_tokens = language_model_instance.embed_tokens
            text_backbone_instance.layers = language_model_instance.layers
            use_gc = getattr(base_model, "is_gradient_checkpointing", False)
            text_backbone_instance.gradient_checkpointing = use_gc
            print("✅ All bridges created successfully.")
            print("--- Patching Text Encoder Internals ---")
            decoder_layers = text_backbone_instance.layers
            FastQwenImagePatcher.apply_patch(base_model)
            for layer in decoder_layers:
                if hasattr(layer.self_attn, 'rotary_emb'):
                    layer.self_attn.rotary_emb.forward = types.MethodType(Qwen2_5_VLRotaryEmbedding_shape_fix, layer.self_attn.rotary_emb)
                layer.input_layernorm.forward = types.MethodType(fast_rms_layernorm, layer.input_layernorm)
                layer.post_attention_layernorm.forward = types.MethodType(fast_rms_layernorm, layer.post_attention_layernorm)
                layer.mlp.forward = types.MethodType(mlp_forward_with_direct_kernel, layer.mlp)
            print("✅ SUCCESS: Patched Text Encoder internals.")

        except Exception as e:
            print(f"❌ CRITICAL: Failure during Qwen2.5-VL post-patching. Error: {e}")
            traceback.print_exc()

        # We can now safely call the base post_patch for final tokenizer fixes, etc.
        return FastLlamaModel.post_patch(model, tokenizer)