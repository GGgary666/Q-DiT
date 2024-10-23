import gc
import torch
import torch.nn as nn
from tqdm import tqdm
import copy
from qdit.qLinearLayer import find_qlinear_layers
from qdit.qMessonic import QuantMeissonicSingleTransformerBlock, QuantMeissonicTransformerBlock
from qdit.gptq import GPTQ, Quantizer_GPTQ
from functools import partial
from models.meissonic import SingleTransformerBlock, TransformerBlock


from .quant import *


def add_act_quant_wrapper(model, device, args, scales):
    transformer_blocks = model.transformer_blocks
    single_transformer_blocks = model.single_transformer_blocks
    
    for i in range(len(transformer_blocks)):
        args_i = copy.deepcopy(args)
        args_i.weight_group_size = args.weight_group_size[i]
        args_i.act_group_size = args.act_group_size[i]
        m = None
        if isinstance(transformer_blocks[i], TransformerBlock):
            m = QuantMeissonicTransformerBlock(
                transormer_block=transformer_blocks[i],
                args=args_i,
            )
        elif isinstance(transformer_blocks[i], QuantMeissonicTransformerBlock):
            m = transformer_blocks[i]

        if m is None:
            continue

        m = m.to(device)

        nameTemplate = 'transformer_blocks.{}.{}.{}'
        m.attn.input_quant.configure(
            partial(quantize_activation_wrapper, args=args_i),
            scales[nameTemplate.format(i, 'attn', 'qkv')]
        )
        m.attn.act_quant.configure(
            partial(quantize_activation_wrapper, args=args_i),
            scales[nameTemplate.format(i, 'attn', 'proj')]
        )
        if args.quantize_bmm_input:
            m.attn.q_quant.configure(
                partial(quantize_attn_q_wrapper, args=args_i),
                None
            )
            m.attn.k_quant.configure(
                partial(quantize_attn_k_wrapper, args=args_i),
                None
            )
            m.attn.v_quant.configure(
                partial(quantize_attn_v_wrapper, args=args_i),
                None
            )
            m.attn.add_k_quant.configure(partial(quantize_attn_add_k_wrapper, args=args_i), None)
            m.attn.add_v_quant.configure(partial(quantize_attn_add_v_wrapper, args=args_i), None)
            m.attn.add_q_quant.configure(partial(quantize_attn_add_q_wrapper, args=args_i), None)

        m.ff.input_quant.configure(
            partial(quantize_activation_wrapper, args=args_i),
            scales[nameTemplate.format(i, 'ff', 'act_fn')]
        )
        m.ff.act_quant.configure(
            partial(quantize_activation_wrapper, args=args_i),
            scales[nameTemplate.format(i, 'ff', 'proj')]
        )
        m.ff_context.input_quant.configure(
            partial(quantize_activation_wrapper, args=args_i),
            scales[nameTemplate.format(i, 'ff_context', 'act_fn')]
        )
        m.ff_context.act_quant.configure(
            partial(quantize_activation_wrapper, args=args_i),
            scales[nameTemplate.format(i, 'ff_context', 'proj')]
        )
        
        transformer_blocks[i] = m.cpu()
        torch.cuda.empty_cache()
        
    for i in range(len(single_transformer_blocks)):
        args_i = copy.deepcopy(args)
        args_i.weight_group_size = args.weight_group_size[i]
        args_i.act_group_size = args.act_group_size[i]
        m = None
        if isinstance(single_transformer_blocks[i], SingleTransformerBlock):
            m = QuantMeissonicSingleTransformerBlock(
                single_transformer_block=single_transformer_blocks[i],
                args=args_i,
            )
        elif isinstance(single_transformer_blocks[i], QuantMeissonicSingleTransformerBlock):
            m = single_transformer_blocks[i]

        if m is None:
            continue

        m = m.to(device)

        nameTemplate = 'single_transformer_block.{}.{}.{}'
        m.attn.input_quant.configure(
            partial(quantize_activation_wrapper, args=args_i),
            scales[nameTemplate.format(i, 'attn', 'qkv')]
        )
        m.attn.act_quant.configure(
            partial(quantize_activation_wrapper, args=args_i),
            scales[nameTemplate.format(i, 'attn', 'proj')]
        )
        if args.quantize_bmm_input:
            m.attn.q_quant.configure(
                partial(quantize_attn_q_wrapper, args=args_i),
                None
            )
            m.attn.k_quant.configure(
                partial(quantize_attn_k_wrapper, args=args_i),
                None
            )
            m.attn.v_quant.configure(
                partial(quantize_attn_v_wrapper, args=args_i),
                None
            )
        
        single_transformer_blocks[i] = m.cpu()
        torch.cuda.empty_cache()
        
    return model

def quantize_model(model, device, args):
    blocks = model.blocks
    for i in tqdm(range(len(blocks))):
        args_i = copy.deepcopy(args)
        args_i.weight_group_size = args.weight_group_size[i]
        args_i.act_group_size = args.act_group_size[i]
        m = None
        if isinstance(blocks[i], DiTBlock):
            m = QuantDiTBlock(
                dit_block=blocks[i],
                args=args_i,
            )
        elif isinstance(blocks[i], QuantDiTBlock):
            m = blocks[i]

        if m is None:
            continue

        m = m.to(device)
        m.mlp.fc1.quant()
        m.mlp.fc2.quant()
        m.attn.qkv.quant()
        m.attn.proj.quant()

        blocks[i] = m.cpu()
        torch.cuda.empty_cache()
    return model

def quantize_layer(model, name, device):
    blocks = model.blocks
    i = int(name.split(".")[1])
    assert(isinstance(blocks[i], QuantDiTBlock))
    m = blocks[i]
    m = m.to(device)

    if name.endswith("mlp.fc1"):
        m.mlp.fc1.quant()
    elif name.endswith("mlp.fc2"):
        m.mlp.fc2.quant()
    elif name.endswith("attn.qkv"):
        m.attn.qkv.quant()
    elif name.endswith("attn.proj"):
        m.attn.proj.quant()
    else:
        raise NotImplementedError

    blocks[i] = m.cpu()
    torch.cuda.empty_cache()
    return model

def quantize_block(block, device):
    assert(isinstance(block, QuantDiTBlock))
    block.to(device)

    block.mlp.fc1.quant()
    block.mlp.fc2.quant()
    block.attn.qkv.quant()
    block.attn.proj.quant()

    torch.cuda.empty_cache()

def quantize_model_gptq(model, device, args, dataloader):
    print('Starting GPTQ quantization ...')
    blocks = model.blocks
    
    quantizers = {}
    for i in tqdm(range(len(blocks))):
        args_i = copy.deepcopy(args)
        args_i.weight_group_size = args.weight_group_size[i]
        args_i.act_group_size = args.act_group_size[i]
        if isinstance(blocks[i], DiTBlock):
            m = QuantDiTBlock(
                dit_block=blocks[i],
                args=args_i,
            )
        elif isinstance(blocks[i], QuantDiTBlock):
            m = blocks[i]
        else:
            continue

        block = m.to(device)

        block_layers = find_qlinear_layers(block)

        sequential = [list(block_layers.keys())]
       
        for names in sequential:
            subset = {n: block_layers[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer_GPTQ()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.w_sym, mse=False, 
                    channel_group=args.weight_channel_group,
                    clip_ratio=args.w_clip_ratio,
                    quant_type=args.quant_type
                )
                
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            
            model.to(device)
            for calib_x, calib_t, calib_y in tqdm(dataloader):
                model(calib_x.to(device), calib_t.to(device), calib_y.to(device))

            for h in handles:
                h.remove()
            
            for name in subset:
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.weight_group_size[0]
                )
                subset[name].quantized = True
                quantizers['model.blocks.%d.%s' % (i, name)] = gptq[name].quantizer.cpu()
                gptq[name].free()

            del gptq

        blocks[i] = block.cpu()
        del block, m
        torch.cuda.empty_cache()
        gc.collect()

    return model
