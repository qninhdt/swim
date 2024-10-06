import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import Encoder, Decoder


class SwimVAE(nn.Module):

    def __init__(self, pretrained_sd: str):
        super(SwimVAE, self).__init__()

        self.sd_vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            pretrained_sd, subfolder="vae"
        )
        self.sd_vae.requires_grad_(False)

        self.sd_vae.encoder.forward = SwimVAE.custom_sd_vae_encoder_fwd.__get__(
            self.sd_vae.encoder, Encoder.__class__
        )
        self.sd_vae.decoder.forward = SwimVAE.custom_sd_vae_decoder_fwd.__get__(
            self.sd_vae.decoder, Decoder.__class__
        )

        # add the skip connection convs
        self.sd_vae.decoder.skip_conv_1 = torch.nn.Conv2d(
            512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
        ).requires_grad_(True)
        self.sd_vae.decoder.skip_conv_2 = torch.nn.Conv2d(
            256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
        ).requires_grad_(True)
        self.sd_vae.decoder.skip_conv_3 = torch.nn.Conv2d(
            128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
        ).requires_grad_(True)
        self.sd_vae.decoder.skip_conv_4 = torch.nn.Conv2d(
            128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
        ).requires_grad_(True)

        # init the skip connection conv weights
        torch.nn.init.constant_(self.sd_vae.decoder.skip_conv_1.weight, 1e-5)
        torch.nn.init.constant_(self.sd_vae.decoder.skip_conv_2.weight, 1e-5)
        torch.nn.init.constant_(self.sd_vae.decoder.skip_conv_3.weight, 1e-5)
        torch.nn.init.constant_(self.sd_vae.decoder.skip_conv_4.weight, 1e-5)

        self.sd_vae.decoder.ignore_skip = False
        self.sd_vae.decoder.gamma = 1

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = (
            self.sd_vae.encode(x).latent_dist.sample()
            * self.sd_vae.config.scaling_factor
        )
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        self.sd_vae.decoder.incoming_skip_acts = self.sd_vae.encoder.current_down_blocks
        x = (self.sd_vae.decode(x / self.sd_vae.config.scaling_factor).sample).clamp(
            -1, 1
        )
        return x

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        trainable_params = []

        trainable_params += list(self.sd_vae.decoder.skip_conv_1.parameters())
        trainable_params += list(self.sd_vae.decoder.skip_conv_2.parameters())
        trainable_params += list(self.sd_vae.decoder.skip_conv_3.parameters())
        trainable_params += list(self.sd_vae.decoder.skip_conv_4.parameters())

        return trainable_params

    @staticmethod
    def custom_sd_vae_encoder_fwd(self: Encoder, sample: torch.Tensor) -> torch.Tensor:
        sample = self.conv_in(sample)
        l_blocks = []
        # down
        for down_block in self.down_blocks:
            l_blocks.append(sample)
            sample = down_block(sample)
        # middle
        sample = self.mid_block(sample)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        self.current_down_blocks = l_blocks
        return sample

    @staticmethod
    def custom_sd_vae_decoder_fwd(
        self: Decoder, sample: torch.Tensor, latent_embeds: torch.Tensor = None
    ) -> torch.Tensor:
        sample = self.conv_in(sample)
        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

        # middle
        sample = self.mid_block(sample, latent_embeds)
        sample = sample.to(upscale_dtype)
        if not self.ignore_skip:
            skip_convs = [
                self.skip_conv_1,
                self.skip_conv_2,
                self.skip_conv_3,
                self.skip_conv_4,
            ]
            # up
            for idx, up_block in enumerate(self.up_blocks):
                skip_in = skip_convs[idx](
                    self.incoming_skip_acts[::-1][idx] * self.gamma
                )
                # add skip
                sample = sample + skip_in
                sample = up_block(sample, latent_embeds)
        else:
            for idx, up_block in enumerate(self.up_blocks):
                sample = up_block(sample, latent_embeds)

        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample
