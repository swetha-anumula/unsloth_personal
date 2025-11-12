# unsloth/models/mmdit_ms_rope.py

import torch
from torch import nn
from typing import Tuple, Optional, Union

# --- Helper for generating tables ---
def _generate_ms_rope_tables(
    seq_len_t: int, 
    seq_len_h: int, 
    seq_len_w: int,
    rotary_dim: int, 
    base_t: float, 
    base_h: float, 
    base_w: float,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, ...]:
    """Generates the 6 cos/sin tables based on their respective bases."""
    
    # RoPE generation logic (assuming same rotary_dim for all axes)
    inv_freq_t = 1.0 / (base_t ** (torch.arange(0, rotary_dim, 2, dtype=torch.int64, device=device).float() / rotary_dim))
    inv_freq_h = 1.0 / (base_h ** (torch.arange(0, rotary_dim, 2, dtype=torch.int64, device=device).float() / rotary_dim))
    inv_freq_w = 1.0 / (base_w ** (torch.arange(0, rotary_dim, 2, dtype=torch.int64, device=device).float() / rotary_dim))
    
    t_idx = torch.arange(seq_len_t, device=device, dtype=torch.int64).float()
    h_idx = torch.arange(seq_len_h, device=device, dtype=torch.int64).float()
    w_idx = torch.arange(seq_len_w, device=device, dtype=torch.int64).float()

    freqs_t, freqs_h, freqs_w = torch.outer(t_idx, inv_freq_t).to(device), torch.outer(h_idx, inv_freq_h).to(device), torch.outer(w_idx, inv_freq_w).to(device)

    emb_t = torch.cat((freqs_t, freqs_t), dim=-1).to(dtype)
    emb_h = torch.cat((freqs_h, freqs_h), dim=-1).to(dtype)
    emb_w = torch.cat((freqs_w, freqs_w), dim=-1).to(dtype)

    cos_t, sin_t = emb_t.cos(), emb_t.sin()
    cos_h, sin_h = emb_h.cos(), emb_h.sin()
    cos_w, sin_w = emb_w.cos(), emb_w.sin()

    return cos_t, sin_t, cos_h, sin_h, cos_w, sin_w

class MMDiT_MS_RotaryEmbedding(nn.Module):
    def __init__(self, rotary_dim: int, config):
        super().__init__()
        
        self.dim = rotary_dim 
        
        # RoPE Base parameters (assuming they are in the model's config)
        self.base_t = getattr(config, "rope_theta_t", 10000.0)
        self.base_h = getattr(config, "rope_theta_h", 10000.0)
        self.base_w = getattr(config, "rope_theta_w", 10000.0)

        # Max sequence lengths (assuming they are T, H, W grid dimensions)
        self.max_len_t = getattr(config, "max_temporal_length", 1) 
        self.max_len_h = getattr(config, "max_height", 32)
        self.max_len_w = getattr(config, "max_width", 32)
        
        self._cos_sin_tables = None # Cache for the tables

    def _get_cos_sin_tables(self, dtype: torch.dtype, device: torch.device):
        if self._cos_sin_tables is not None:
            # Ensure tables are on the correct device/dtype if necessary
            if self._cos_sin_tables[0].dtype != dtype or self._cos_sin_tables[0].device != device:
                 self._cos_sin_tables = [t.to(dtype=dtype, device=device) for t in self._cos_sin_tables]
            return self._cos_sin_tables
        
        self._cos_sin_tables = _generate_ms_rope_tables(
            self.max_len_t, self.max_len_h, self.max_len_w,
            self.dim, self.base_t, self.base_h, self.base_w,
            dtype, device
        )
        return self._cos_sin_tables


    def forward(self, x: torch.Tensor, position_ids: Optional[torch.LongTensor] = None) -> Tuple[torch.Tensor, ...]:
        """Returns the 7 tensors for the Triton kernel."""
        B, H, L, D = x.shape
        device = x.device
        dtype = x.dtype
        
        # 1. Generate/Load the 6 cos/sin tables
        cos_t, sin_t, cos_h, sin_h, cos_w, sin_w = self._get_cos_sin_tables(dtype, device)
        
        # 2. Prepare Positional Indices (pt, ph, pw)
        if position_ids is None:
            current_t, current_h, current_w = self.max_len_t, self.max_len_h, self.max_len_w
            
            # Create 3D meshgrid of indices and flatten
            w_indices = torch.arange(current_w, device=device)
            h_indices = torch.arange(current_h, device=device)
            t_indices = torch.arange(current_t, device=device)

            p_w, p_h, p_t = torch.meshgrid(w_indices, h_indices, t_indices, indexing='ij')
            
            # Stack and reshape to (T*H*W, 3) -> (L, 3)
            pos_indices_L3 = torch.stack((p_t.reshape(-1), p_h.reshape(-1), p_w.reshape(-1)), dim=-1)
            
            # Replicate for the batch: (B*L, 3)
            pos_indices = pos_indices_L3.repeat(B, 1).to(torch.int32)
            
        else:
            # If position_ids is provided externally, flatten it.
            pos_indices = position_ids.reshape(B * L, 3).to(torch.int32)

        # Return (indices, cos/sin_t, cos/sin_h, cos/sin_w)
        return (pos_indices, cos_t, sin_t, cos_h, sin_h, cos_w, sin_w)