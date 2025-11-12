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
# Import fast RoPE kernel for Text Encoder (standard 1D RoPE)
from unsloth.models.llama import inplace_rope_embedding 
# NEW IMPORT: Import the MS-RoPE kernel (to make MMDiT forward work when called)
from unsloth.kernels.ms_rope import fast_ms_rope_embedding 


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
    #print("#⚡ M1 ATTENTION KEYHOLE RUNNING.")
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
    # This calls the fast 1D RoPE kernel
    query_states, key_states = inplace_rope_embedding(query_states, key_states, cos, sin, position_ids)
    
    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    past_key_value = (key_states, value_states) if use_cache else None

    # --- M1 FIXED CONDITION ---
    is_causal_mask_check = attention_mask is None or \
                           (attention_mask.ndim == 4 and attention_mask.shape[2] == 1)

    if HAS_FLASH_ATTENTION and is_causal_mask_check:
        #print("flash_attn_enabled")
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
        
        attn_output = attn_output.transpose(1, 2)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output_final = self.o_proj(attn_output)
    
    # --- FIX THE RETURN SIGNATURE FOR THE QWEN LAYER ---
    # The Text Encoder's layer forward is failing, expecting 2 values.
    return attn_output_final, None 


def MMDiTAttention_keyhole_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    raise RuntimeError("MMDiT Patches disabled for Encoder-Only test.")


class FastQwenImagePatcher:
    """
    Patches the Qwen2_5_VLForConditionalGeneration module (the text encoder).
    """
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

# NEW MMDiT Attention Forward (for the MMDiT layers)
def MMDiTAttention_ms_rope_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    # 1. Standard Q/K/V Projection
    bsz, q_len, _ = hidden_states.size()
    
    # --- FIX 1: Use correct projection names (to_q, to_k, to_v) ---
    query_states = self.to_q(hidden_states)
    key_states = self.to_k(hidden_states)
    value_states = self.to_v(hidden_states)
    
    # --- FIX 2: Use correct dimension names and calculate head_dim ---
    num_heads = self.heads
    head_dim = self.to_q.out_features // num_heads # Calculated from projection output
    num_key_value_heads = getattr(self, 'num_key_value_heads', num_heads) # Assume GQA/MQA is handled

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    # 2. MS-RoPE Calculation
    ms_rope_tensors = self.rotary_emb(query_states, position_ids=position_ids)

    # Unpack the 7 tensors: (B*L, 3) indices, 6 tables (L, D)
    pos_indices, cos_t, sin_t, cos_h, sin_h, cos_w, sin_w = ms_rope_tensors
    
    # 3. Apply Fast MS-RoPE Kernel (PASS-THROUGH FIX: This is now a dummy function)
    query_states, key_states = fast_ms_rope_embedding(
        query_states, 
        key_states, 
        pos_indices, 
        cos_t, sin_t, cos_h, sin_h, cos_w, sin_w
    )
    
    # 4. Standard Attention Logic (FlashAttention/Standard)
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    past_key_value = (key_states, value_states) if use_cache else None

    is_causal_mask_check = attention_mask is None or \
                           (attention_mask.ndim == 4 and attention_mask.shape[2] == 1)

    num_key_value_groups = num_heads // num_key_value_heads
    attn_output = None # Initialize for UnboundLocalError fix
    
    if HAS_FLASH_ATTENTION and is_causal_mask_check:
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
        
        attn_output = attn_output.transpose(1, 2)

    # Final reshaping requires hidden_size, which is total dim
    hidden_size = num_heads * head_dim
    
    # FIX: Ensure attn_output is not None before reshaping
    if attn_output is None:
        raise RuntimeError("Attention output was not computed.") 
    o_proj_in_features = self.o_proj.in_features
    
    # Reshape the attention output to match the input dimension of the final projection layer.
    attn_output = attn_output.reshape(bsz, q_len, o_proj_in_features)
    
    # Now, project it. The output of this will have the correct final dimension.
    attn_output_final = self.o_proj(attn_output)

    # (You can keep the debug prints from Step 2 for now to verify the fix)
    print(f"[DEBUG Attention Out Fix] Final output shape: {attn_output_final.shape}")
    if attn_output_final.shape[-1] != 3840:
         print("!!! WARNING: FIX FAILED, SHAPE IS STILL WRONG !!!")
    
    return attn_output_final, None
