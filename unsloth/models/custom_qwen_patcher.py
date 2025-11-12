import types
import torch
from typing import Optional, Tuple
from peft import get_peft_model as _get_peft_model, LoraConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTENTION = True
except ImportError:
    print("WARNING: flash_attn is not installed. Custom kernel will have no effect.")
    HAS_FLASH_ATTENTION = False
import types
import torch
from typing import Optional, Tuple
from unsloth.models.llama import inplace_rope_embedding


def Qwen2_5_VLAttention_keyhole_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    #print("keyhole detected")
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if position_ids is None:
        position_ids = torch.arange(
            kv_seq_len - q_len,
            kv_seq_len,
            dtype=torch.long,
            device=hidden_states.device
        ).unsqueeze(0)

    cos, sin = self.rotary_emb(
        query_states,
        position_ids=position_ids
    )
    query_states, key_states = inplace_rope_embedding(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    past_key_value = (key_states, value_states) if use_cache else None

    is_causal_mask_check = attention_mask is None or \
                           (attention_mask.ndim == 4 and attention_mask.shape[2] == 1)

    attn_output = None

    if HAS_FLASH_ATTENTION and is_causal_mask_check:
        attn_output = flash_attn_func(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            causal=True
        )
    else:
        if self.num_key_value_groups != 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

    o_proj_in_features = self.o_proj.in_features
    attn_output = attn_output.reshape(bsz, q_len, o_proj_in_features)
    attn_output_final = self.o_proj(attn_output)

    return attn_output_final, None


'''def MMDiTAttention_keyhole_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    raise RuntimeError("MMDiT Patches disabled for Encoder-Only test.")'''


class FastQwenImagePatcher:
    @staticmethod
    def apply_patch(model):
        print("\n--- Applying KEYHOLE SURGICAL Patch ---")
        patched_layers = 0
        try:
            decoder_layers = model.model.language_model.layers
            for layer in decoder_layers:
                attention_module = layer.self_attn
                if isinstance(attention_module, Qwen2_5_VLAttention):
                    attention_module.forward = types.MethodType(Qwen2_5_VLAttention_keyhole_forward, attention_module)
                    patched_layers += 1
            print(f"✅ SUCCESS: Surgically patched {patched_layers} attention blocks with keyhole kernel.")
        except Exception as e:
            import traceback
            print(f"❌ CRITICAL FAILURE: Failed during surgical patch: {e}")
            traceback.print_exc()
        return model


'''def MMDiTAttention_ms_rope_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    bsz, q_len, _ = hidden_states.size()

    is_cross_attention = encoder_hidden_states is not None
    
    query_states = self.to_q(hidden_states)

    if is_cross_attention:
        key_states = self.to_k(encoder_hidden_states)
        value_states = self.to_v(encoder_hidden_states)
        kv_seq_len = encoder_hidden_states.shape[-2]
    else: # Self-Attention
        key_states = self.to_k(hidden_states)
        value_states = self.to_v(hidden_states)
        kv_seq_len = hidden_states.shape[-2]

    num_heads = self.heads
    head_dim = self.to_q.out_features // num_heads
    num_key_value_heads = getattr(self, 'num_key_value_heads', num_heads)

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, kv_seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, kv_seq_len, num_key_value_heads, head_dim).transpose(1, 2)

    # MS-RoPE is skipped as it conflicts with the QwenImageTransformer2DModel's architecture.

    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    past_key_value = (key_states, value_states) if use_cache else None

    is_causal_mask_check = attention_mask is None or \
                           (attention_mask.ndim == 4 and attention_mask.shape[2] == 1)

    num_key_value_groups = num_heads // num_key_value_heads
    attn_output = None

    if HAS_FLASH_ATTENTION and is_causal_mask_check and not is_cross_attention:
        attn_output = flash_attn_func(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            causal=True
        )
    else:
        if num_key_value_groups != 1:
            key_states = key_states.repeat_interleave(num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

    if attn_output is None:
        raise RuntimeError("Attention output was not computed.")

    hidden_size = num_heads * head_dim
    attn_output = attn_output.reshape(bsz, q_len, hidden_size)

    attn_output_processed = self.to_out[0](attn_output)
    return (attn_output_processed, attn_output_processed)'''
