from typing import List

import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from transformers import AutoTokenizer, CLIPTextModel
from safetensors.torch import load_model, save_model

from .swim_unet import SwimUnet
from .swim_vae import SwimVAE


class Swim(nn.Module):

    def __init__(
        self,
        pretrained_sd: str,
        revision: str,
        lora_rank_unet: int,
    ):
        super(Swim, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_sd,
            subfolder="tokenizer",
            revision=revision,
            use_fast=False,
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_sd, subfolder="text_encoder"
        )
        self.text_encoder.requires_grad_(False)

        self.vae = SwimVAE(pretrained_sd)
        self.unet = SwimUnet(pretrained_sd, lora_rank_unet)

        self.noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
            pretrained_sd, subfolder="scheduler"
        )
        self.n_timesteps = self.noise_scheduler.config.num_train_timesteps

    def get_trainable_parameters(self):
        return [
            *self.unet.parameters(),
            *self.vae.parameters(),
        ]

    def compute_text_embeddings(self, texts: list[str]) -> torch.Tensor:
        tokens: torch.Tensor = self.tokenizer(
            texts,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        tokens = tokens.to(self.text_encoder.device)

        embeddings = self.text_encoder(tokens)[0]

        return embeddings

    def forward(
        self, x: torch.Tensor, text_input: list[str] | torch.Tensor
    ) -> torch.Tensor:
        if isinstance(text_input, list):
            text_embeddings = self.compute_text_embeddings(text_input)
        else:
            text_embeddings = text_input

        B = x.shape[0]
        timesteps = [self.n_timesteps - 1] * B

        return self.forward_with_embeddings(
            x,
            text_embeddings,
            timesteps,
        )

    def forward_with_embeddings(
        self, x: torch.Tensor, text_embeddings: torch.Tensor, timesteps: List[int]
    ) -> torch.Tensor:
        B = x.shape[0]
        encoded_x = self.vae.encode(x)

        packed_timesteps = torch.tensor(timesteps, dtype=torch.long, device=x.device)

        model_pred = self.unet.forward(
            encoded_x,
            packed_timesteps,
            encoder_hidden_states=text_embeddings,
        )

        # add noise
        encoded_x = torch.stack(
            [
                self.noise_scheduler.step(
                    model_pred[i], timesteps[i], encoded_x[i], return_dict=True
                ).prev_sample
                for i in range(B)
            ]
        )

        decoded_x = self.vae.decode(encoded_x)

        return decoded_x

    def unload_text_encoder(self):
        del self.text_encoder
        del self.tokenizer
        torch.cuda.empty_cache()
