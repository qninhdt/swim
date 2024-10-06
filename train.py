import torch
from models.swim import Swim
from accelerate import init_empty_weights
from torchinfo import summary
from time import time


def get_device(module):
    return next(module.parameters()).device


model = Swim("stabilityai/sd-turbo", None, 8).to("cpu")

sample = torch.randn(1, 3, 512, 512, device="cpu")
latent = torch.randn(1, 4, 64, 64, device="cpu")
input_text_embedding = torch.randn(1, 77, 1024, device="cpu")
packed_timesteps = torch.tensor([1], dtype=torch.long, device="cpu")


def num_params(module):
    return sum(p.numel() for p in module.parameters())


model.unload_text_encoder()

# model.unet = torch.compile(model.unet, mode="reduce-overhead", fullgraph=True)


# print("Number of trainable parameters:", num_params(model))
# summary(model, input_data=(sample, input_text_embedding))
summary(model.unet, input_data=(latent, packed_timesteps, input_text_embedding))
# def step(model: Swim, sample, input_text_embedding):
#     last = time()
#     # forward pass
#     output = model.forward(sample, input_text_embedding)
#     # backward pass
#     output.mean().backward()
#     print("Time:", time() - last)


# # before compiling
# step(model, sample, input_text_embedding)

# model.unet = torch.compile(model.unet, mode="reduce-overhead", fullgraph=True)

# # after compiling
# for _ in range(5):
#     step(model, sample, input_text_embedding)
