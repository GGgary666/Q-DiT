import torch
from torch import nn
from typing import List, Optional, Tuple
import math
from .quant import Quantizer, quantize_tensor, quantize_tensor_channel_group
from .qLinearLayer import QLinearLayer
# from models.models import DiTBlock, modulate, TimestepEmbedder

from models.meissonic import TransformerBlock, SingleTransformerBlock
from models.attention import Attention, FeedForward
from models.embeddings import apply_rotary_emb
from models.activations import GELU
from models.normalization import AdaLayerNormZeroSingle, AdaLayerNormZero
import torch.nn.functional as F
from copy import deepcopy

    

class QuantAttention(nn.Module):
    """
    multimodal cross attention
    Args:
        nn (_type_): _description_
    """
    def __init__(
            self,
            attn: Attention,
            args
            ) -> None:
        super().__init__()
        # 量化参数
        self.quantize_bmm_input = args.quantize_bmm_input
        # self.abits = args.abits
        # 模型结构
        self.query_dim = attn.query_dim
        self.is_cross_attention = attn.cross_attention_dim
        self.cross_attention_dim = attn.cross_attention_dim
        self.added_kv_proj_dim = attn.added_kv_proj_dim
        self.dim_head = attn.dim_head
        self.use_bias = attn.use_bias
        self.heads = attn.heads
        self.context_pre_only = attn.context_pre_only
        self.scale = attn.scale
        self.pre_only = attn.pre_only
        self.dropout = attn.dropout
        
        self.norm_q = attn.norm_q
        self.norm_k = attn.norm_k
       
        self.to_q = QLinearLayer(attn.to_q, deepcopy(args))
        self.to_k = QLinearLayer(attn.to_k, deepcopy(args))
        self.to_v = QLinearLayer(attn.to_v, deepcopy(args))
        
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = QLinearLayer(attn.add_k_proj, deepcopy(args))
            self.add_v_proj = QLinearLayer(attn.add_v_proj, deepcopy(args))
            if self.context_pre_only is not None:
                self.add_q_proj = QLinearLayer(attn.add_q_proj, deepcopy(args))

        if not self.pre_only:
            self.to_out = nn.ModuleList([])
            self.to_out_proj = QLinearLayer(attn.to_out_proj, deepcopy(args))
            self.to_out.append(self.to_out_proj)
            self.to_out.append(nn.Dropout(self.dropout))
        
        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_add_out = QLinearLayer(attn.to_add_out, deepcopy(args))
    
        self.norm_added_q = attn.norm_added_q
        self.norm_added_k = attn.norm_added_k
        
        self.input_quant = Quantizer(args=deepcopy(args))
        if self.quantize_bmm_input:
            self.q_quant = Quantizer(args=deepcopy(args))
            self.k_quant = Quantizer(args=deepcopy(args))
            self.v_quant = Quantizer(args=deepcopy(args))
            if self.added_kv_proj_dim is not None:
                self.add_k_quant = Quantizer(args=deepcopy(args))
                self.add_v_quant = Quantizer(args=deepcopy(args))
                if self.context_pre_only is not None:
                    self.add_q_quant = Quantizer(args=deepcopy(args))
        self.act_quant = Quantizer(args=deepcopy(args))
        
        # self.register_buffer("reorder_index_qkv", None)
        # self.register_buffer("reorder_index_proj", None)
        
    def to(self, *args, **kwargs):
        super(QuantAttention, self).to(*args, **kwargs)
        self.to_q = self.to_q.to(*args, **kwargs)
        self.to_k = self.to_k.to(*args, **kwargs)
        self.to_v = self.to_v.to(*args, **kwargs)
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = self.add_k_proj.to(*args, **kwargs)
            self.add_v_proj = self.add_v_proj.to(*args, **kwargs)
            if self.context_pre_only is not None:
                self.add_q_proj = self.add_q_proj.to(*args, **kwargs)
        if not self.pre_only:
            self.to_out = self.to_out.to(*args, **kwargs)
        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_add_out = self.to_add_out.to(*args, **kwargs)
        if self.quantize_bmm_input:
            self.q_quant = self.q_quant.to(*args, **kwargs)
            self.k_quant = self.k_quant.to(*args, **kwargs)
            self.v_quant = self.v_quant.to(*args, **kwargs)
            if self.added_kv_proj_dim is not None:
                self.add_k_quant = self.add_k_quant.to(*args, **kwargs)
                self.add_v_quant = self.add_v_quant.to(*args, **kwargs)
                if self.context_pre_only is not None:
                    self.add_q_quant =  self.add_q_quant.to(*args, **kwargs)
        self.input_quant = self.input_quant.to(*args, **kwargs)
        self.act_quant = self.act_quant.to(*args, **kwargs)
        return self
    
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
            # quant operator for input activation
            hidden_states = self.input_quant(hidden_states)
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
            # attention input quant operator
            if self.quantize_bmm_input:
                query = self.q_quant(query)
                key = self.k_quant(key)
                value = self.v_quant(value)
            # quant operator for input activation
            encoder_hidden_states = self.input_quant(encoder_hidden_states)
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
            if self.quantize_bmm_input:
                encoder_hidden_states_query_proj = self.add_q_quant(encoder_hidden_states_query_proj)
                encoder_hidden_states_key_proj = self.add_k_quant(encoder_hidden_states_key_proj)
                encoder_hidden_states_value_proj = self.add_v_quant(encoder_hidden_states_value_proj)
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
            # output activation quant operator
            hidden_states = self.act_quant(hidden_states)
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
            hidden_states = self.input_quant(hidden_states)
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
            # attention input quant operator
            if self.quantize_bmm_input:
                query = self.q_quant(query)
                key = self.k_quant(key)
                value = self.v_quant(value)
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * self.dim_head)
            hidden_states = hidden_states.to(query.dtype)
            # output activation quant operator
            hidden_states = self.act_quant(hidden_states)
            return hidden_states

class QuantGELU(nn.Module):
    r"""
    GELU activation function with tanh approximation support with `approximate="tanh"`.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        approximate (`str`, *optional*, defaults to `"none"`): If `"tanh"`, use tanh approximation.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, 
                gelu_net : GELU,
                args):
        super().__init__()
        self.proj = QLinearLayer(gelu_net.proj, deepcopy(args))
        self.approximate = gelu_net.approximate

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.device.type != "mps":
            return F.gelu(gate, approximate=self.approximate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32), approximate=self.approximate).to(dtype=gate.dtype)

    def to(self, *args, **kwargs):
        super(QuantGELU, self).to(*args, **kwargs)
        self.proj = self.proj.to(*args, **kwargs)
        self.approximate = self.approximate.to(*args, **kwargs)
        return self
    
    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states
    
class QuantFeedForward(nn.Module):
    def __init__(
        self,
        ffn: FeedForward,
        args
    ):
        super().__init__()
        self.act_fn = QuantGELU(ffn.act_fn, deepcopy(args))
        self.dropout = ffn.dropout
        self.final_dropout = ffn.final_dropout
        self.input_quant = Quantizer(args=deepcopy(args))
        self.act_quant = Quantizer(args=deepcopy(args))
        self.net = nn.ModuleList([])
        # input quant
        self.net.append(self.input_quant)
        # project in
        self.net.append(self.act_fn)
        # project dropout
        self.net.append(nn.Dropout(self.dropout))
        # act quant
        self.net.append(self.act_quant)
        # project out
        self.proj = QLinearLayer(ffn.proj, deepcopy(args))
        self.net.append(self.proj)
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if self.final_dropout:
            self.net.append(nn.Dropout(self.dropout))
        
    def to(self, *args, **kwargs):
        super(QuantFeedForward, self).to(*args, **kwargs)
        self.net = self.net.to(*args, **kwargs)
        return self
    
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states

class QuantAdaLayerNormZeroSingle(nn.Module):
    def __init__(
        self, 
        adalnzs : AdaLayerNormZeroSingle,
        args
        ):
        super().__init__()
        self.silu = adalnzs.silu
        self.linear = QLinearLayer(adalnzs.linear, deepcopy(args))
        self.norm = adalnzs.norm
    
    def to(self, *args, **kwargs):
        super(QuantAdaLayerNormZeroSingle, self).to(*args, **kwargs)
        self.silu = self.silu.to(*args, **kwargs)
        self.linear = self.linear.to(*args, **kwargs)
        self.norm = self.norm.to(*args, **kwargs)
        return self

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa

class QuantAdaLayerNormZero(nn.Module):
    def __init__(
        self, 
        adalnz : AdaLayerNormZero,
        args
        ):
        super().__init__()
        self.emb = adalnz.emb
        self.silu = adalnz.SiLU()
        self.linear = QLinearLayer(adalnz.linear, deepcopy(args))
        self.norm = adalnz.norm
    
    def to(self, *args, **kwargs):
        super(QuantAdaLayerNormZero, self).to(*args, **kwargs)
        self.emb = self.emb.to(*args, **kwargs)
        self.silu = self.silu.to(*args, **kwargs)
        self.linear = self.linear.to(*args, **kwargs)
        self.norm = self.norm.to(*args, **kwargs)
        return self
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class QuantMeissonicSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        single_tranformer_block: SingleTransformerBlock,
        args
    ):
        super().__init__()
        self.args = args
        self.quantize_bmm_input = args.quantize_bmm_input
        # self.mlp_hidden_dim = single_tranformer_block.mlp_hidden_dim
        self.norm = QuantAdaLayerNormZeroSingle(single_tranformer_block.norm, deepcopy(args))
        self.proj_mlp = QLinearLayer(single_tranformer_block.proj_mlp, deepcopy(args))
        self.act_mlp = single_tranformer_block.act_mlp
        self.proj_out = QLinearLayer(single_tranformer_block.proj_out, deepcopy(args))
        self.attn = QuantAttention(single_tranformer_block.attn, deepcopy(args))
        if self.quantize_bmm_input:
            self.norm_quant = Quantizer(args=deepcopy(args))
            self.proj_mlp_quant = Quantizer(args=deepcopy(args))
            self.attn_quant = Quantizer(args=deepcopy(args))
            self.proj_out_quant = Quantizer(args=deepcopy(args))
    
    def to(self, *args, **kwargs):
        super(QuantMeissonicSingleTransformerBlock, self).to(*args, **kwargs)
        self.norm = self.norm.to(*args, **kwargs)
        self.proj_mlp = self.proj_mlp.to(*args, **kwargs)
        self.act_mlp = self.act_mlp.to(*args, **kwargs)
        self.proj_out = self.proj_out.to(*args, **kwargs)
        self.attn = self.attn.to(*args, **kwargs)
        if self.quantize_bmm_input:
            self.norm_quant = self.norm_quant.to(*args, **kwargs)
            self.proj_mlp_quant = self.proj_mlp_quant.to(*args, **kwargs)
            self.attn_quant = self.attn_quant.to(*args, **kwargs)
            self.proj_out_quant = self.proj_out_quant.to(*args, **kwargs)
        return self
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
    ):
        residual = hidden_states
        if not self.quantize_bmm_input:
            norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
            mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
            attn_output = self.attn(
                hidden_states=norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
            )
            hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
            gate = gate.unsqueeze(1)
            hidden_states = gate * self.proj_out(hidden_states)
            hidden_states = residual + hidden_states
            if hidden_states.dtype == torch.float16:
                hidden_states = hidden_states.clip(-65504, 65504)
            return hidden_states
        else:
            norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
            norm_hidden_states = self.norm_quant(norm_hidden_states)
            mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
            mlp_hidden_states = self.proj_mlp_quant(mlp_hidden_states)
            attn_output = self.attn_quant(self.attn(
                hidden_states=norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
            ))
            
            hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
            gate = gate.unsqueeze(1)
            hidden_states = gate * self.proj_out_quant(self.proj_out(hidden_states))
            hidden_states = residual + hidden_states
            if hidden_states.dtype == torch.float16:
                hidden_states = hidden_states.clip(-65504, 65504)
            return hidden_states

class QuantMeissonicTransformerBlock(nn.Module):
    def __init__(
        self,
        transormer_block: TransformerBlock,
        args
    ):
        super().__init__()
        self.args = args
        self.quantize_bmm_input = args.quantize_bmm_input
        
        self.norm1 = transormer_block.norm1
        self.norm1_context = transormer_block.norm1_context
        self.attn = QuantAttention(transormer_block.attn, deepcopy(args))
        self.norm2 = transormer_block.norm2
        self.ff = transormer_block.ff
        self.norm2_context = transormer_block.norm2_context
        self.ff_context = transormer_block.ff_context
        if self.quantize_bmm_input:
            self.norm1_quant = Quantizer(args=deepcopy(args))
            self.norm1_context_quant = Quantizer(args=deepcopy(args))
            self.attn_quant = Quantizer(args=deepcopy(args))
            self.context_atnn_quant = Quantizer(args=deepcopy(args))
            self.norm2_quant = Quantizer(args=deepcopy(args))
            self.ff_quant = Quantizer(args=deepcopy(args))
            self.norm2_context_quant = Quantizer(args=deepcopy(args))
            self.ff_context_quant = Quantizer(args=deepcopy(args))
            
    def to(self, *args, **kwargs):
        super(QuantMeissonicTransformerBlock, self).to(*args, **kwargs)
        self.norm1 = self.norm1.to(*args, **kwargs)
        self.norm1_context = self.norm1_context.to(*args, **kwargs)
        self.attn = self.attn.to(*args, **kwargs)
        self.norm2 = self.norm2 .to(*args, **kwargs)
        self.ff = self.ff.to(*args, **kwargs)
        self.norm2_context = self.norm2_context.to(*args, **kwargs)
        self.ff_context = self.ff_context.to(*args, **kwargs)
        if self.quantize_bmm_input:
            self.norm1_quant = self.norm1_quant.to(*args, **kwargs)
            self.norm1_context_quant = self.norm1_context_quant.to(*args, **kwargs)
            self.attn_quant = self.attn_quant.to(*args, **kwargs)
            self.context_atnn_quant = self.context_atnn_quant.to(*args, **kwargs)
            self.norm2_quant = self.norm2_quant.to(*args, **kwargs)
            self.ff_quant = self.ff_quant.to(*args, **kwargs)
            self.norm2_context_quant = self.norm2_context_quant.to(*args, **kwargs)
            self.ff_context_quant = self.ff_context_quant.to(*args, **kwargs)
        return self
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
    ):
        if not self.quantize_bmm_input:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb
            )
            # Attention.
            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
            )
            # Process attention outputs for the `hidden_states`.
            attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = hidden_states + attn_output
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            ff_output = self.ff(norm_hidden_states)
            ff_output = gate_mlp.unsqueeze(1) * ff_output
            hidden_states = hidden_states + ff_output
            # Process attention outputs for the `encoder_hidden_states`.
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output
            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
            if encoder_hidden_states.dtype == torch.float16:
                encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
            return encoder_hidden_states, hidden_states
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
            norm_hidden_states = self.norm1_quant(norm_hidden_states)
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb
            )
            norm_encoder_hidden_states = self.norm1_context_quant(norm_encoder_hidden_states)
            # Attention.
            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
            )
            attn_output = self.attn_quant(attn_output)
            context_attn_output = self.context_atnn_quant(attn_output)
            # Process attention outputs for the `hidden_states`.
            attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = hidden_states + attn_output
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = self.norm2_quant(norm_hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            ff_output = self.ff(norm_hidden_states)
            ff_output = self.ff_quant(ff_output)
            ff_output = gate_mlp.unsqueeze(1) * ff_output
            hidden_states = hidden_states + ff_output
            # Process attention outputs for the `encoder_hidden_states`.
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output
            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = self.norm2_context_quant(norm_encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            context_ff_output = self.ff_context(norm_encoder_hidden_states)
            context_ff_output = self.ff_context_quant(context_ff_output)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
            if encoder_hidden_states.dtype == torch.float16:
                encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
            return encoder_hidden_states, hidden_states