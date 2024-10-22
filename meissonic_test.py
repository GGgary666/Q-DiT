# import os
# import sys
# sys.path.append("./")

import torch


from safetensors.torch import load_file
import json
from models.meissonic import Transformer2DModel

model_path = "/home/dongpeijie/.cache/huggingface/hub/models--MeissonFlow--Meissonic/snapshots/08ff13de62d55a6984806076d005089acc63f9ee/transformer/"
config_name = "config.json"
file_name = "diffusion_pytorch_model.safetensors"

loaded = load_file(model_path + file_name)

# 打开一个包含JSON数据的文件
with open(model_path + config_name, 'r') as file:
    # 使用json.load()将文件中的JSON数据解码为Python对象
    model_config_dict = json.load(file)

meissonic_mdoel = Transformer2DModel(
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

meissonic_mdoel.load_state_dict(loaded)

print(meissonic_mdoel)



# Set the manual seed for reproducibility
torch.manual_seed(42)

# Generate tensors based on the provided sizes and types
hidden_states = torch.randint(0, 100, (2, 64, 64), dtype=torch.int64)
micro_conds = torch.randn(2, 5, dtype=torch.float32)
pooled_projections = torch.randn(2, 1024, dtype=torch.float32)
encoder_hidden_states = torch.randn(2, 77, 1024, dtype=torch.float32)
img_ids = torch.randint(0, 100, (1024, 3), dtype=torch.int64)
txt_ids = torch.randn(77, 3, dtype=torch.float32)
timestep = torch.randint(0, 10, (1,), dtype=torch.int64)

# Print the generated tensors to verify
# print("hidden_states:", hidden_states)
# print("micro_conds:", micro_conds)
# print("pooled_projections:", pooled_projections)
# print("encoder_hidden_states:", encoder_hidden_states)
# print("img_ids:", img_ids)
# print("txt_ids:", txt_ids)
# print("timestep:", timestep)

cur_res = meissonic_mdoel(
    hidden_states = hidden_states,
    micro_conds=micro_conds,
    pooled_projections=pooled_projections,
    encoder_hidden_states=encoder_hidden_states,
    img_ids = img_ids,
    txt_ids = txt_ids,
    timestep = timestep,
)