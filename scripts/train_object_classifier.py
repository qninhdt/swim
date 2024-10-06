from typing import Literal

import torch
import timm
import click
import json
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb

SUPPORTED_CLASSESS = ["pedestrian", "car", "bus", "truck"]


class BDD100KDataset(Dataset):

    def __init__(self, path: str):
        with open(os.path.join(path, "labels.json"), "r") as f:
            self.labels = json.load(f)

        self.path = path

    def __len__(self):
        return len(self.labels) // 100

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = Image.open(os.path.join(self.path, "images", label["name"]))
        image = transforms.ToTensor()(image)

        category = SUPPORTED_CLASSESS.index(label["category"])

        return {
            "image": image,
            "category": category,
            "timeofday": label["timeofday"],
            "weather": label["weather"],
        }


def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    categories = torch.tensor([item["category"] for item in batch])
    timeofday = [item["timeofday"] for item in batch]
    weather = [item["weather"] for item in batch]

    return {
        "image": images,
        "category": categories,
        "timeofday": timeofday,
        "weather": weather,
    }


class Accuracy:

    def __init__(self):
        self.total = 0
        self.correct = 0

    def update(self, output, target):
        self.total += 1
        self.correct += int(output.argmax() == target)

    def compute(self):
        return self.correct / self.total

    def reset(self):
        self.total = 0
        self.correct = 0


@click.command()
@click.option("--model_name", type=str, default="resnet50")
@click.option("--dataset", type=str, required=True)
@click.option("--batch_size", type=int, default=64)
@click.option("--num_epochs", type=int, default=10)
@click.option("--lr", type=float, default=1e-3)
@click.option("--weight_decay", type=float, default=1e-4)
@click.option("--log_interval", type=int, default=100)
@click.option("--save_interval", type=int, default=1)
@click.option("--save_dir", type=str, default="checkpoints")
@click.option("--num_workers", type=int, default=4)
@click.option("--seed", type=int, default=42)
def train_object_classifier(
    model_name: str,
    dataset: str,
    batch_size: int,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    log_interval: int,
    save_interval: int,
    save_dir: str,
    num_workers: int,
    seed: int,
):
    wandb.init(project="bdd100k_object_classification")

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = timm.create_model(
        model_name, pretrained=True, num_classes=len(SUPPORTED_CLASSESS)
    )
    model.train()
    model.cuda()

    train_dataset = BDD100KDataset(os.path.join(dataset, "train"))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    val_dataset = BDD100KDataset(os.path.join(dataset, "val"))
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    train_accuracy_map = {
        "all": Accuracy(),
        **{category: Accuracy() for category in SUPPORTED_CLASSESS},
    }
    val_accuracy_map = {
        "all": Accuracy(),
        **{category: Accuracy() for category in SUPPORTED_CLASSESS},
    }

    for epoch in range(num_epochs):

        model.train()

        for i, batch in tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch {epoch}",
        ):
            images = batch["image"].cuda()
            categories = batch["category"].cuda()

            optimizer.zero_grad()

            one_hot_gt = torch.zeros(
                (categories.size(0), len(SUPPORTED_CLASSESS)), device=categories.device
            )
            one_hot_gt[range(categories.size(0)), categories] = 1
            output = model(images)
            loss = criterion(output, one_hot_gt)

            loss.backward()
            optimizer.step()

            for x, y in zip(output, categories):
                train_accuracy_map["all"].update(x, y)
                train_accuracy_map[SUPPORTED_CLASSESS[y.item()]].update(x, y)

            wandb.log({"loss": loss.item()})

            if i % log_interval == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss {loss.item()}")

        if epoch % save_interval == 0:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(
                model.state_dict(), os.path.join(save_dir, f"{model_name}_{epoch}.pth")
            )
            print(f"Model saved at {save_dir}/{model_name}_{epoch}.pth")

        # evaluate model
        model.eval()

        with torch.no_grad():

            for batch in val_dataloader:
                images = batch["image"].cuda()
                categories = batch["category"].cuda()

                output = model(images)

                for x, y in zip(output, categories):
                    val_accuracy_map["all"].update(x, y)
                    val_accuracy_map[SUPPORTED_CLASSESS[y.item()]].update(x, y)

        print("Train accuracy:")
        print(f"-- all: {train_accuracy_map['all'].compute()}")
        for category in SUPPORTED_CLASSESS:
            print(
                f"Epoch {epoch}, -- {category} {train_accuracy_map[category].compute()}"
            )

        print("Validation accuracy:")
        print(f"-- all: {val_accuracy_map['all'].compute()}")
        for category in SUPPORTED_CLASSESS:
            print(
                f"Epoch {epoch}, -- {category} {val_accuracy_map[category].compute()}"
            )

        wandb.log(
            {
                "train/acc": train_accuracy_map["all"].compute(),
                "val/acc": val_accuracy_map["all"].compute(),
            }
        )

        for name, acc in train_accuracy_map.items():
            wandb.log({f"train/{name}_acc": acc.compute()})

        for name, acc in val_accuracy_map.items():
            wandb.log({f"val/{name}_acc": acc.compute()})

        for acc in train_accuracy_map.values():
            acc.reset()

        for acc in val_accuracy_map.values():
            acc.reset()


if __name__ == "__main__":
    train_object_classifier()
