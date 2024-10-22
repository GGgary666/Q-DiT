"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import torch
import logging
from PIL import Image
from pytorch_lightning import seed_everything
from tqdm import tqdm
import math

from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from utils.download import find_model
from models.models import DiT_models
from utils.logger_setup import create_logger
from glob import glob
from copy import deepcopy

from qdit.quant import *
from qdit.outlier import *
from qdit.datautils import *
from collections import defaultdict
from qdit.modelutils import quantize_model, quantize_model_gptq,  add_act_quant_wrapper


from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from safetensors.torch import load_file
import json


from diffusers import VQModel
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from models.meissonic import Transformer2DModel
from meissonic_pipeline.scheduler import Scheduler
from meissonic_pipeline.pipeline import Pipeline

def initialize_tranformer_model(
    model_path : str = "/home/dongpeijie/.cache/huggingface/hub/models--MeissonFlow--Meissonic/snapshots/08ff13de62d55a6984806076d005089acc63f9ee/transformer/",
    config_name : str = "config.json",
    file_name : str = "diffusion_pytorch_model.safetensors",
    device : str = "device",
    ):
    # load model config
    with open(model_path + config_name, 'r') as file:
        # 使用json.load()将文件中的JSON数据解码为Python对象
        model_config_dict = json.load(file)
    # initialize model structure
    meissonic_model = Transformer2DModel(
        patch_size=model_config_dict['patch_size'],
        in_channels=model_config_dict['in_channels'],
        num_layers=model_config_dict['num_layers'],
        num_single_layers=model_config_dict['num_single_layers'],
        attention_head_dim=model_config_dict['attention_head_dim'],
        num_attention_heads=model_config_dict['num_attention_heads'],
        joint_attention_dim=model_config_dict['joint_attention_dim'],
        pooled_projection_dim= model_config_dict['pooled_projection_dim'],
        guidance_embeds=model_config_dict['guidance_embeds'], # unused in our implementation
        axes_dims_rope=tuple(model_config_dict['axes_dims_rope']),
        vocab_size=model_config_dict['vocab_size'],
        codebook_size=model_config_dict['codebook_size'],
        downsample=model_config_dict['downsample'],
        upsample=model_config_dict['upsample'],
    )
    loaded_weight = load_file(model_path + file_name)
    meissonic_model.load_state_dict(loaded_weight)
    return meissonic_model.to(device)

def initialize_meissonic_pipeline(
    tranformer_model : Transformer2DModel,
    model_path : str = "MeissonFlow/Meissonic",
    text_encoder_path : str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    device : str = 'cuda',
    ):
    vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae", )
    # text_encoder = CLIPTextModelWithProjection.from_pretrained(model_path,subfolder="text_encoder",)
    #using original text enc for stable sampling
    text_encoder = CLIPTextModelWithProjection.from_pretrained(text_encoder_path) 
    tokenizer = CLIPTokenizer.from_pretrained(model_path,subfolder="tokenizer",)
    scheduler = Scheduler.from_pretrained(model_path,subfolder="scheduler",)
    pipe=Pipeline(vq_model, tokenizer=tokenizer,text_encoder=text_encoder,transformer=tranformer_model,scheduler=scheduler)
    return pipe.to(device)
    

def validate_model(args, transformer_model):
    """
    用于验证模型生成图像的质量
    """
    seed_everything(args.seed)
    device = next(transformer_model.parameters()).device
    # 初始化 meissonic pipeline
    pipe = initialize_meissonic_pipeline(tranformer_model=transformer_model, device=device)
    # 准备输入的测试数据
    prompts = [
        "Two actors are posing for a pictur with one wearing a black and white face paint.",
        "A large body of water with a rock in the middle and mountains in the background.",
        "A white and blue coffee mug with a picture of a man on it.",
        "A statue of a man with a crown on his head.",
        "A man in a yellow wet suit is holding a big black dog in the water.",
        "A white table with a vase of flowers and a cup of coffee on top of it.",
        "A woman stands on a dock in the fog.",
        "A woman is standing next to a picture of another woman."
    ]
    # 目前只考虑了batchsize = 1
    # batched_generation = (args.batch_size != 1)
    num_images = len(prompts) # if batched_generation else 1
    negative_prompt = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"
    # 向模型喂入数据，并保存输出图像
    for i in range(num_images):
        images = pipe(
            prompt=prompts[i:i+1], 
            negative_prompt=[negative_prompt],
            height=args.image_size,
            width=args.image_size,
            guidance_scale=args.cfg_scale,
            num_inference_steps=args.num_sampling_steps,
            output_type = "pil",
        ).images
        image = images[0]
        sanitized_prompt = prompts[i][:6].replace(" ", "_")
        save_path = os.path.join(args.experiment_dir, f"{sanitized_prompt}_{args.image_size}_{args.num_sampling_steps}_{args.cfg_scale}.png")
        image.save(save_path)
        logging.info(f"The {i+1}/{num_images} image is saved to {save_path}")
    print("Finish validating samples!")
    
    
# def create_npz_from_sample_folder(sample_dir, num=50_000):
#     """
#     该函数从图像文件夹中读取图像，并将它们保存为一个.npz文件
#     """
#     samples = []
#     for i in tqdm(range(num), desc="Building .npz file from samples"):
#         sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
#         sample_np = np.asarray(sample_pil).astype(np.uint8)
#         samples.append(sample_np)
#     samples = np.stack(samples)
#     assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
#     npz_path = f"{sample_dir}.npz"
#     np.savez(npz_path, arr_0=samples)
#     print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
#     return npz_path

# def sample_fid(args, model, diffusion, vae):
#     """
#     该函数生成大量图像样本，并将它们保存为.png文件
#     """
#     # Create folder to save samples:
#     seed_everything(args.seed)
#     device = next(model.parameters()).device
#     using_cfg = args.cfg_scale > 1.0
#     model_string_name = args.model.replace("/", "-")
#     ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
#     folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
#                   f"cfg-{args.cfg_scale}-seed-{args.seed}"
#     sample_folder_dir = f"{args.experiment_dir}/{folder_name}"
#     os.makedirs(sample_folder_dir, exist_ok=True)
#     print(f"Saving .png samples at {sample_folder_dir}")

#     # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
#     n = args.batch_size
#     # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
#     total_samples = int(math.ceil(args.num_fid_samples / n) * n)
#     print(f"Total number of images that will be sampled: {total_samples}")
#     iterations = int(total_samples // n)
#     pbar = range(iterations)
#     pbar = tqdm(pbar)
#     total = 0
#     for _ in pbar:
#         # Sample inputs:
#         z = torch.randn(n, model.in_channels, model.input_size, model.input_size, device=device)
#         y = torch.randint(0, args.num_classes, (n,), device=device)

#         # Setup classifier-free guidance:
#         if using_cfg:
#             z = torch.cat([z, z], 0)
#             y_null = torch.tensor([1000] * n, device=device)
#             y = torch.cat([y, y_null], 0)
#             model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
#         else:
#             model_kwargs = dict(y=y)

#         z = z.half()
#         with autocast():
#             samples = diffusion.ddim_sample_loop(
#                 model, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
#             )
#         if using_cfg:
#             samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

#         samples = vae.decode(samples / 0.18215).sample
#         samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

#         # Save samples to disk as individual .png files
#         for i, sample in enumerate(samples):
#             index = i + total
#             Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
#         total += n

#     create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
#     print("Done.")


def main():
    args = create_argparser().parse_args()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'device: {device}')
    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    quant_method = "qdit"
    quant_string_name = f"{quant_method}_w{args.wbits}a{args.abits}"
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{quant_string_name}"  # Create an experiment folder
    args.experiment_dir = experiment_dir
    os.makedirs(experiment_dir, exist_ok=True)
    create_logger(experiment_dir)
    logging.info(f"Experiment directory created at {experiment_dir}")
    logging.info(f"""wbits: {args.wbits}, abits: {args.abits}, w_sym: {args.w_sym}, a_sym: {args.a_sym},
                 weight_group_size: {args.weight_group_size}, act_group_size: {args.act_group_size},
                 quant_method: {args.quant_method}, use_gptq: {args.use_gptq}, static: {args.static},
                 image_size: {args.image_size}, cfg_scale: {args.cfg_scale}""")
    
    # Load model:
    
    tranformer_model = initialize_tranformer_model(device=device)
    tranformer_model.eval()  # important!
    # diffusion = create_diffusion(str(args.num_sampling_steps))
    # vae = AutoencoderKL.from_pretrained(f"../sd-vae-ft-{args.vae}").to(device)
    args.weight_group_size = eval(args.weight_group_size)
    args.act_group_size = eval(args.act_group_size)
    args.weight_group_size = [args.weight_group_size] * len(model.blocks) if isinstance(args.weight_group_size, int) else args.weight_group_size
    args.act_group_size = [args.act_group_size] * len(model.blocks) if isinstance(args.act_group_size, int) else args.act_group_size
    
    print("Inserting activations quantizers ...")
    
    # 需要重写get_act_scales 和 add_act_quant_wrapper函数
    
    # if args.static:
    #     dataloader = get_loader(args.calib_data_path, nsamples=1024, batch_size=16)
    #     print("Getting activation stats...")
    #     scales = get_act_scales(
    #         model, diffusion, dataloader, device, args
    #     )
    # else:
    #     scales = defaultdict(lambda: None)
    # model = add_act_quant_wrapper(model, device=device, args=args, scales=scales)

    print("Quantizing ...")
    if args.use_gptq:
        dataloader = get_loader(args.calib_data_path, nsamples=256)
        model = quantize_model_gptq(model, device=device, args=args, dataloader=dataloader)
    else:
        model = quantize_model(model, device=device, args=args)

    print("Finish quant!")
    logging.info(model)

    # generate some sample images
    model.to(device)
    model.half()
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    torch.set_grad_enabled(False)
    validate_model(args, model)
    
    # sample_fid(args, model, diffusion, vae)

def create_argparser():
    parser = argparse.ArgumentParser()

    # quantization parameters
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 5, 6, 8, 16],
        help='#bits to use for quantizing weight; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--abits', type=int, default=16, choices=[2, 3, 4, 5, 6, 8, 16],
        help='#bits to use for quantizing activation; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--exponential', action='store_true',
        help='Whether to use exponent-only for weight quantization.'
    )
    parser.add_argument(
        '--quantize_bmm_input', action='store_true',
        help='Whether to perform bmm input activation quantization. Default is not.'
    )
    parser.add_argument(
        '--a_sym', action='store_true',
        help='Whether to perform symmetric quantization. Default is asymmetric.'
    )
    parser.add_argument(
        '--w_sym', action='store_true',
        help='Whether to perform symmetric quantization. Default is asymmetric.'
    )
    parser.add_argument(
        '--static', action='store_true',
        help='Whether to perform static quantization (For activtions). Default is dynamic. (Deprecated in Atom)'
    )
    parser.add_argument(
        '--weight_group_size', type=str,
        help='Group size when quantizing weights. Using 128 as default quantization group.'
    )
    parser.add_argument(
        '--weight_channel_group', type=int, default=1,
        help='Group size of channels that will quantize together. (only for weights now)'
    )
    parser.add_argument(
        '--act_group_size', type=str,
        help='Group size when quantizing activations. Using 128 as default quantization group.'
    )
    parser.add_argument(
        '--tiling', type=int, default=0, choices=[0, 16],
        help='Tile-wise quantization granularity (Deprecated in Atom).'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--use_gptq', action='store_true',
        help='Whether to use GPTQ for weight quantization.'
    )
    parser.add_argument(
        '--quant_method', type=str, default='max', choices=['max', 'mse'],
        help='The method to quantize weight.'
    )
    parser.add_argument(
        '--a_clip_ratio', type=float, default=1.0,
        help='Clip ratio for activation quantization. new_max = max * clip_ratio'
    )
    parser.add_argument(
        '--w_clip_ratio', type=float, default=1.0,
        help='Clip ratio for weight quantization. new_max = max * clip_ratio'
    )
    parser.add_argument(
        '--save_dir', type=str, default='../saved',
        help='Path to store the reordering indices and quantized weights.'
    )
    parser.add_argument(
        '--quant_type', type=str, default='int', choices=['int', 'fp'],
        help='Determine the mapped data format by quant_type + n_bits. e.g. int8, fp4.'
    )
    parser.add_argument(
        '--calib_data_path', type=str, default='../cali_data/cali_data_256.pth',
        help='Path to store the reordering indices and quantized weights.'
    )
    # Inherited from DiT
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512, 1024], default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=9.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--results-dir", type=str, default="../results")
    parser.add_argument(
        "--save_ckpt", action="store_true", help="choose to save the qnn checkpoint"
    )
    # sample_ddp.py
    parser.add_argument("--tf32", action="store_true",
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    return parser


if __name__ == "__main__": 
    main()
