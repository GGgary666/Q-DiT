from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalization import RMSNorm
from .embeddings import apply_rotary_emb
from .activations import GELU

class Attention(nn.Module):
    """
    multimodal cross attention
    Args:
        nn (_type_): _description_
    """
    def __init__(
            self,
            query_dim: int,  # q维度
            cross_attention_dim: Optional[int] = None, # 如果为none，cross_attention_dim = query_dim
            added_kv_proj_dim: Optional[int] = None,
            dim_head: Optional[int] = None,
            heads: Optional[int] = None,
            out_dim: Optional[int] = None,
            context_pre_only: Optional[bool] = None, # 不知道这是啥
            bias: Optional[bool] = True,
            qk_norm: Optional[str] = "rms_norm",
            eps: Optional[float] = 1e-6,
            pre_only: Optional[bool] = False,
            dropout: Optional[float] = 0.0,
            out_bias: Optional[bool] = True,
            added_proj_bias: Optional[bool] = True,
            ) -> None:
        super().__init__()
        assert heads * dim_head == query_dim, 'query dim should be divisible by num_heads'
        self.query_dim = query_dim
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.added_kv_proj_dim = added_kv_proj_dim
        self.dim_head = dim_head
        self.use_bias = bias
        self.heads = heads
        self.context_pre_only = context_pre_only
        self.scale = self.dim_head ** -0.5
        self.pre_only = pre_only
        self.dropout = dropout
        
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        elif qk_norm == "layer_norm":
            self.norm_q = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=True)
            self.norm_k = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=True)
        
        self.to_q = nn.Linear(query_dim, query_dim, bias=bias)
        self.to_k = nn.Linear(self.cross_attention_dim, query_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim, query_dim, bias=bias)
        
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, query_dim, bias=added_proj_bias)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, query_dim, bias=added_proj_bias)
            if self.context_pre_only is not None:
                self.add_q_proj = nn.Linear(added_kv_proj_dim, query_dim, bias=added_proj_bias)

        if not self.pre_only:
            self.to_out = nn.ModuleList([])
            self.to_out_proj = nn.Linear(query_dim, query_dim, bias=out_bias)
            self.to_out.append(self.to_out_proj)
            self.to_out.append(nn.Dropout(self.dropout))
        
        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_add_out = nn.Linear(query_dim, query_dim, bias=out_bias)
        
        if qk_norm is not None and added_kv_proj_dim is not None:
            if qk_norm == "rms_norm":
                self.norm_added_q = RMSNorm(dim_head, eps=eps)
                self.norm_added_k = RMSNorm(dim_head, eps=eps)
            else:
                self.norm_added_q = None
                self.norm_added_k = None
        
    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.FloatTensor = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
        if encoder_hidden_states is not None:
            batch_size, _, _ = encoder_hidden_states.shape
            # `sample` projections.
            query = self.to_q(hidden_states)
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)
            query = query.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
            key = key.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
            value = value.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
            if self.norm_q is not None:
                query = self.norm_q(query)
            if self.norm_k is not None:
                key = self.norm_k(key)
            encoder_hidden_states_query_proj = self.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, self.heads, self.dim_head
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, self.heads, self.dim_head
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, self.heads, self.dim_head
            ).transpose(1, 2)
            if self.norm_added_q is not None:
                encoder_hidden_states_query_proj = self.norm_added_q(encoder_hidden_states_query_proj)
            if self.norm_added_k is not None:
                encoder_hidden_states_key_proj = self.norm_added_k(encoder_hidden_states_key_proj)
            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * self.dim_head)
            hidden_states = hidden_states.to(query.dtype)
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states

        else:
            batch_size, _, _ = hidden_states.shape
            # `sample` projections.
            query = self.to_q(hidden_states)
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)
            query = query.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
            key = key.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
            value = value.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
            if self.norm_q is not None:
                query = self.norm_q(query)
            if self.norm_k is not None:
                key = self.norm_k(key)
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * self.dim_head)
            hidden_states = hidden_states.to(query.dtype)
            return hidden_states
            

class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        self.dim_out = dim_out if dim_out is not None else dim
        self.dropout = dropout
        self.final_dropout = final_dropout
        if activation_fn == "gelu":
            self.act_fn = GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            self.act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
    
        self.net = nn.ModuleList([])
        # project in
        self.net.append(self.act_fn)
        # project dropout
        self.net.append(nn.Dropout(self.dropout))
        # project out
        self.proj = nn.Linear(inner_dim, dim_out, bias=bias)
        self.net.append(self.proj)
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if self.final_dropout:
            self.net.append(nn.Dropout(self.dropout))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states

