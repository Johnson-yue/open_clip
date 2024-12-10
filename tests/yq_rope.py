# Copied from Diffusers: https://github.com/huggingface/diffusers/blob/074e12358bc17e7dbe111ea4f62f05dbae8a49d5/src/diffusers/models/embeddings.py#L713C1-L777C1
# Copied from Messionic: https://github.com/viiika/Meissonic/blob/0d9a5481292b1592acc2925c0b90726fa64fe8a5/src/transformer.py#L194C1-L248C25
from typing import List, Optional, Tuple, Union
import torch.nn as nn
import torch
from torch.nn import functional as F
import einops
import  numpy as np
from open_clip.model import CLIP
from open_clip.transformer import text_global_pool, ResidualAttentionBlock # for add rope in TextTransformer
from open_clip.transformer import _expand_token ,VisionTransformer    # for add rope in VisionTransformer
import types


def get_2d_rotary_pos_embed(embed_dim, crops_coords, grid_size, use_real=True):
    """
    RoPE for image tokens with 2d structure.

    Args:
    embed_dim: (`int`):
        The embedding dimension size
    crops_coords (`Tuple[int]`)
        The top-left and bottom-right coordinates of the crop.
    grid_size (`Tuple[int]`):
        The grid size of the positional embedding.
    use_real (`bool`):
        If True, return real part and imaginary part separately. Otherwise, return complex numbers.

    Returns:
        `torch.Tensor`: positional embedding with shape `( grid_size * grid_size, embed_dim/2)`.
    """
    start, stop = crops_coords
    grid_h = np.linspace(start[0], stop[0], grid_size[0], endpoint=False, dtype=np.float32)
    grid_w = np.linspace(start[1], stop[1], grid_size[1], endpoint=False, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)  # [2, W, H]

    grid = grid.reshape([2, 1, *grid.shape[1:]])
    pos_embed = get_2d_rotary_pos_embed_from_grid(embed_dim, grid, use_real=use_real)
    return pos_embed


def get_2d_rotary_pos_embed_from_grid(embed_dim, grid, use_real=False):
    assert embed_dim % 4 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_rotary_pos_embed(
        embed_dim // 2, grid[0].reshape(-1), use_real=use_real
    )  # (H*W, D/2) if use_real else (H*W, D/4)
    emb_w = get_1d_rotary_pos_embed(
        embed_dim // 2, grid[1].reshape(-1), use_real=use_real
    )  # (H*W, D/2) if use_real else (H*W, D/4)

    if use_real:
        cos = torch.cat([emb_h[0], emb_w[0]], dim=1)  # (H*W, D)
        sin = torch.cat([emb_h[1], emb_w[1]], dim=1)  # (H*W, D)
        return cos, sin
    else:
        emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D/2)
        return emb


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            the dtype of the frequency tensor.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    theta = theta * ntk_factor
    freqs = (
        1.0
        / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device)[: (dim // 2)] / dim))
        / linear_factor
    )  # [D/2]
    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    if use_real and repeat_interleave_real:
        # flux, hunyuan-dit, cogvideox
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        # stable audio, allegro
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        n_unsqueeze = x.ndim - 2
        for i in range(n_unsqueeze):
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2) # -2 is last dim
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)
    

#############################################
# Add rope to TextEmbedding on first layer and all Attention Block
# Copied from :
#   src/open_clip/model.py : CLIP.encode_text()
#############################################
def add_rope_to_text_embedding(self:CLIP, text, normalize: bool = False, return_feat=False):
    cast_dtype = self.transformer.get_cast_dtype()

    x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

    x_rope_emb = get_1d_rotary_pos_embed(
        dim=x.size(-1),     #  d_model
        pos=x.size(-2),     #  seq_len == n_ctx 
        repeat_interleave_real=True, use_real=True, freqs_dtype=torch.float64
    ) # cos, sin
    k_x_rope = apply_rotary_emb(x, x_rope_emb)
    x = k_x_rope.to(cast_dtype)
    # x = x + self.positional_embedding.to(cast_dtype)
    
    x = self.transformer(x, attn_mask=self.attn_mask)
    x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
    if return_feat:
        return x
    x, _ = text_global_pool(x, text, self.text_pool_type)
    if self.text_projection is not None:
        if isinstance(self.text_projection, nn.Linear):
            x = self.text_projection(x)
        else:
                x = x @ self.text_projection

    return F.normalize(x, dim=-1) if normalize else x

def add_rope_to_att_qk(self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,):
    # Norm and Rope
    q_x_rope_embed = get_1d_rotary_pos_embed(
        dim=q_x.size(-1), 
        pos=q_x.size(-2),     # or [np.array([0,1, 2, 3, 4])]
        repeat_interleave_real=True, use_real=True, freqs_dtype=torch.float64
    ) # cos, sin
    q_x_rope = apply_rotary_emb(self.ln_1(q_x), q_x_rope_embed)

    k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
    if k_x is not None:
        k_x_rope_embed =  get_1d_rotary_pos_embed(
            dim=k_x.size(-1), 
            pos=k_x.size(-2),     # or [np.array([0,1, 2, 3, 4])]
            repeat_interleave_real=True, use_real=True, freqs_dtype=torch.float64
        ) # cos, sin
        k_x = apply_rotary_emb(k_x, k_x_rope_embed)
    v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
    attn_out = self.attention(q_x=q_x_rope, k_x=k_x, v_x=v_x, attn_mask=attn_mask)
    x = q_x + self.ls_1(attn_out)
    # residual is not position embedding information, so use q_x not q_rope
    x = x + self.ls_2(self.mlp(self.ln_2(x)))
    return x


#############################################
# Add rope to VisionEmbedding on first layer and All Attention block
# Copied from :
#   src/open_clip/transformer.py : VisionTransformer.forward()
#############################################
def add_rope_to_vision_embedding(self:VisionTransformer, x: torch.Tensor):
    x = self.conv1(x)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

    # class embeddings and positional embeddings
    class_embedding = _expand_token(self.class_embedding, x.shape[0]).to(x.dtype)
    
    # Add Rope to embedding
    cls_rotary2d_emb = get_2d_rotary_pos_embed(class_embedding.size(-1), ((0,0),(0,0)), 
                                               grid_size=(1, 1), use_real=True )
    x_rotary2d_emb = get_2d_rotary_pos_embed(x.size(-1), ((1,1), (1+self.grid_size[0],1+self.grid_size[1])),
                                             grid_size=self.grid_size, use_real=True)
    rotary2d_emb = (
        torch.cat([cls_rotary2d_emb[0], x_rotary2d_emb[0]], dim=0).to(x.dtype),
        torch.cat([cls_rotary2d_emb[1], x_rotary2d_emb[1]], dim=0).to(x.dtype)
    )
    
    x = torch.cat([class_embedding, x], dim=1)
    # shape = [*, 1 + grid ** 2 , width]

    # x = x + self.positional_embedding.to(x.dtype)
    x = apply_rotary_emb(x, rotary2d_emb)

    x = self.patch_dropout(x)
    x = self.ln_pre(x)
    x = self.transformer(x)

    if self.attn_pool is not None:
        if self.attn_pool_contrastive is not None:
            # This is untested, WIP pooling that should match paper
            x = self.ln_post(x)  # TBD LN first or separate one after each pool?
            tokens = self.attn_pool(x)
            if self.attn_pool_type == 'parallel':
                pooled = self.attn_pool_contrastive(x)
            else:
                assert self.attn_pool_type == 'cascade'
                pooled = self.attn_pool_contrastive(tokens)
        else:
            # this is the original OpenCLIP CoCa setup, does not match paper
            x = self.attn_pool(x)
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)
    elif self.final_ln_after_pool:
        pooled, tokens = self._global_pool(x)
        pooled = self.ln_post(pooled)
    else:
        x = self.ln_post(x)
        pooled, tokens = self._global_pool(x)

    if self.proj is not None:
        pooled = pooled @ self.proj

    if self.output_tokens:
        return pooled, tokens
        
    return pooled

def add_rope_to_vision_att_qk(self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,):
    # In Vision Attention Block, it received  feature from image 
    # shape is (bs, 1+grid_size**2, seq_dim)
    # Norm and Rope
    grid_size = int(np.sqrt(q_x.size(1)- 1))
    seq_dim = q_x.size(-1)

    cls_rotary2d_emb = get_2d_rotary_pos_embed(
        seq_dim,((0,0), (1,1)),
        grid_size=(1,1), use_real=True
    )
    image_rotary2d_emb = get_2d_rotary_pos_embed(
        seq_dim, ((1,1), (1+grid_size, 1+grid_size)),
        grid_size=(grid_size, grid_size), use_real=True
    )
    rotary2d_emb = (
        torch.cat([cls_rotary2d_emb[0], image_rotary2d_emb[0]], dim=0),
        torch.cat([cls_rotary2d_emb[1], image_rotary2d_emb[1]], dim=0)
    )

    q_rope = apply_rotary_emb(self.ln_1(q_x), rotary2d_emb)

    k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
    if k_x is not None:
        k_x = apply_rotary_emb(k_x, rotary2d_emb) 
    v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
    x = q_x + self.ls_1(self.attention(q_x=q_rope, k_x=k_x, v_x=v_x, attn_mask=attn_mask))
    # residual is not position embedding information, so use q_x not q_rope
    x = x + self.ls_2(self.mlp(self.ln_2(x)))
    return x



###################################
# API function
##################################
def add_rope_to_model(model:CLIP):
    
    def replace_attention_forward(module, new_fn):
        for name, child in module.named_children():
            if isinstance(child, ResidualAttentionBlock):
                child.forward = types.MethodType(new_fn, child)
            else:
                replace_attention_forward(child, new_fn)
    

    # 1) Add rope to first text_embeddings
    model.encode_text = add_rope_to_text_embedding.__get__(model)

    # 1.2) Add rope to  attn layers in TextTransformer
    replace_attention_forward(model.transformer, add_rope_to_att_qk)

    # 2.1)  Add 2d rope to first ViT_embeddings
    model.visual.forward = add_rope_to_vision_embedding.__get__(model.visual)

    # 2.2) Add 2d rope to attn layers in ViT
    replace_attention_forward(model.visual, add_rope_to_vision_att_qk)
    

    return model

if __name__ == "__main__":

    # q = torch.randn(1, 8, 5, 10)  # (batch, heads, seq_len, seq_dim)
    # k = torch.randn(1, 8, 5, 10)  # (batch, heads, seq_len, seq_dim) 

    # Inference code for 1-D rope
    image_rotary_emb = get_1d_rotary_pos_embed(
        dim=q.size(-1), 
        pos=q.size(-2),     # or [np.array([0,1, 2, 3, 4])]
        repeat_interleave_real=True, use_real=True, freqs_dtype=torch.float64
    ) # cos, sin

    mession_q = apply_rotary_emb(q, image_rotary_emb)
