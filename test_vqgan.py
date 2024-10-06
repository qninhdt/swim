import torch
from diffusers import AutoencoderKL, AsymmetricAutoencoderKL
from PIL import Image
from torch import autocast
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
import torchvision.transforms as T

vae: AutoencoderKL = AutoencoderKL.from_pretrained(
    "stabilityai/sd-turbo", subfolder="vae"
)
# vae = AsymmetricAutoencoderKL.from_pretrained(
#     "cross-attention/asymmetric-autoencoder-kl-x-2"
# )
vae.eval()
vae.requires_grad_(False)
vae = vae.cuda()

ori_image = Image.open("car.png")
ori_image = T.Compose([T.Resize(512), T.CenterCrop(512)])(ori_image)

ori_image.save("car_cropped.png")

ori_image = pil_to_tensor(ori_image).cuda()
image = ((ori_image / 255) * 2 - 1).unsqueeze(0)

ori_image = ori_image.permute(1, 2, 0)

latent = vae.encode(image).latent_dist.sample()

reconstructed_image = vae.decode(latent).sample
reconstructed_image = reconstructed_image.squeeze(0)

# save the reconstructed image
reconstructed_image = (reconstructed_image + 1) / 2
reconstructed_image = reconstructed_image.clamp(0, 1)
reconstructed_image = reconstructed_image.mul(255).byte().squeeze(0).permute(1, 2, 0)
Image.fromarray(reconstructed_image.cpu().numpy()).save("reconstructed_car.png")

# save difference image
diff_image = (ori_image - reconstructed_image).abs()
Image.fromarray(diff_image.cpu().numpy()).save("diff_car.png")
# mse loss
mse_loss = (
    ori_image.float() - reconstructed_image.float()
).abs().sum() / ori_image.numel()

print(f"MSE Loss: {mse_loss.item()}")
