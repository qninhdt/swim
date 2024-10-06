import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoTokenizer, CLIPModel


class CLIPScore(nn.Module):

    def __init__(self, model="openai/clip-vit-base-patch32"):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.processor = AutoProcessor.from_pretrained(model)

    def to_cuda(self, inputs):
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def get_image_feature(self, images):
        feat = self.model.get_image_features(
            **self.to_cuda(self.processor(images=images, return_tensors="pt"))
        )
        feat /= feat.norm(dim=-1, keepdim=True)
        return feat

    def get_text_feature(self, texts):
        feat = self.model.get_text_features(
            **self.to_cuda(self.tokenizer(texts, padding=True, return_tensors="pt"))
        )
        feat /= feat.norm(dim=-1, keepdim=True)
        return feat

    def forward(self, img1, text1, img2, text2):
        img1_feat = self.get_image_feature(img1)
        img2_feat = self.get_image_feature(img2)
        text1_feat = self.get_text_feature(text1)
        text2_feat = self.get_text_feature(text2)

        img_dir = img1_feat - img2_feat
        img_dir /= img_dir.norm(dim=-1, keepdim=True)

        text_dir = text1_feat - text2_feat
        text_dir /= text_dir.norm(dim=-1, keepdim=True)

        clip_score1 = torch.einsum("bz,bz->b", img1_feat, text1_feat)
        clip_score2 = torch.einsum("bz,bz->b", img2_feat, text2_feat)
        dclip_score = torch.einsum("bz,bz->b", img_dir, text_dir)

        return clip_score1.item(), clip_score2.item(), dclip_score.item()


from PIL import Image

print(
    CLIPScore()(
        Image.open("./cropped_bdd100k/val/images/000000.jpg"),
        "a car",
        Image.open("./cropped_bdd100k/val/images/000053.jpg"),
        "a person",
    )
)
