# /workspace/swetha/unsloth/unsloth/models/common_patches.py
# FINAL CORRECTED VERSION

import torch
import types
from typing import Optional, Tuple, Union, List
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLRotaryEmbedding

# ====================================================================
# <<< THE DEFINITIVE FIX: Import the REAL, high-performance kernel >>>
# This replaces the slow, local pure-PyTorch version with the actual
# Triton-powered kernel from the unsloth library.
from unsloth.kernels import fast_rms_layernorm
# ====================================================================


# <<< WE HAVE DELETED the old, slow `fast_rms_layernorm` function >>>
# def fast_rms_layernorm(self, X, gemma = False):
#    ... (The old code is now gone) ...


torch_nn_functional_silu = torch.nn.functional.silu
torch_nn_functional_linear = torch.nn.functional.linear

# This MLP/FFN patch is fine as it is. It uses standard PyTorch ops
# that autograd can handle, and it provides a speed benefit through fusion.
def fast_swiglu_forward(self, X):
    gate = torch_nn_functional_linear(X, self.gate_proj.weight, self.gate_proj.bias)
    up   = torch_nn_functional_linear(X, self.up_proj.weight, self.up_proj.bias)
    gate = torch_nn_functional_silu(gate, inplace = True)
    gate *= up 
    down = torch_nn_functional_linear(gate, self.down_proj.weight, self.down_proj.bias)
    
    return down

# The rest of your file is correct and can remain as is.
def Qwen2_5_VLRotaryEmbedding_shape_fix(self, x: torch.Tensor, position_ids: torch.LongTensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    if hasattr(self, "inv_freq"):
        dim = self.inv_freq.shape[0] * 2
    else:
        dim = x.shape[-1] 
    base = self.base if hasattr(self, 'base') else self.config.rope_theta 
    if position_ids is not None:
        seq_len = position_ids.max().item() + 1
    else:
        seq_len = x.shape[-2] 

    dtype = x.dtype
    device = x.device
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
    )
    
    t = torch.arange(seq_len, device=device, dtype=torch.int64).float()
    if hasattr(self, "scaling_factor") and self.scaling_factor != 1.0:
        t = t / self.scaling_factor
    
    freqs = torch.outer(t, inv_freq).to(device)
    emb = torch.cat((freqs, freqs), dim=-1).to(dtype)
    cos_table = emb.cos()
    sin_table = emb.sin()
    return cos_table, sin_table

def LlamaDecoderLayer_fast_forward(
    self,
    hidden_states:       torch.Tensor,
    causal_mask          = None,
    attention_mask:      Optional[torch.Tensor] = None,
    position_ids:        Optional[torch.LongTensor] = None,
    past_key_value:      Optional[Tuple[torch.Tensor]] = None,
    output_attentions:   Optional[bool] = False,
    use_cache:           Optional[bool] = False,
    padding_mask:        Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    *args, **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    
    residual = hidden_states
    
    # This call will now resolve to the REAL Triton kernel
    hidden_states = fast_rms_layernorm(self.input_layernorm, hidden_states) 
    
    attn_outputs = self.self_attn(
        hidden_states       = hidden_states,
        causal_mask         = causal_mask,
        attention_mask      = attention_mask,
        position_ids        = position_ids,
        past_key_value      = past_key_value,
        output_attentions   = output_attentions, 
        use_cache           = use_cache,
        padding_mask        = padding_mask,
        position_embeddings = position_embeddings,
    )
    if len(attn_outputs) == 3:
        hidden_states, self_attn_weights, present_key_value = attn_outputs
    elif len(attn_outputs) == 2:
        hidden_states, present_key_value = attn_outputs
        self_attn_weights = None 
    else:
        raise ValueError(f"Attention module returned {len(attn_outputs)} values; expected 2 or 3.")
    hidden_states = residual + hidden_states
    residual = hidden_states
    
    # This call will now also resolve to the REAL Triton kernel
    hidden_states = fast_rms_layernorm(self.post_attention_layernorm, hidden_states)
    
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    outputs = (hidden_states,)
    if output_attentions: outputs += (self_attn_weights,) 
    if use_cache: outputs += (present_key_value,)
    
    return outputs