import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from diffusers.models.autoencoders.vae import Encoder, Decoder
from peft import LoraConfig


class SwimUnet(nn.Module):

    def __init__(self, pretrained_sd: str, lora_rank: int):
        super(SwimUnet, self).__init__()

        self.sd_unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            pretrained_sd, subfolder="unet"
        )
        self.sd_unet.requires_grad_(False)

        sd_unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )

        self.sd_unet.add_adapter(sd_unet_lora_config)

    def get_trainable_parameters(self):
        return list(filter(lambda p: p.requires_grad, self.sd_unet.parameters()))

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.sd_unet.forward(x, timesteps, encoder_hidden_states).sample
