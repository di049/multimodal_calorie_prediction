import torch
from torch.utils.data import Dataset

from PIL import Image
import timm
import numpy as np
import pandas as pd

from transformers import AutoTokenizer

import albumentations as A

seed = 42


class MultimodalDataset(Dataset):

    def __init__(self, config, transforms, ds_type="train"):
        if ds_type == "train":
            self.df = pd.read_csv(config.DF_PATH)
            self.df = self.df[self.df['split'] == 'train'].reset_index()
        else:
            self.df = pd.read_csv(config.DF_PATH)
            self.df = self.df[self.df['split'] == 'test'].reset_index()
        self.image_cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        ingredients = self.df.loc[idx, "ingredients"]
        total_calories = self.df.loc[idx, "total_calories"]

        img_path = self.df.loc[idx, "dish_id"]
        try:
            image = Image.open(f"data/images/{img_path}.png").convert('RGB')
        except:
            image = torch.randint(0, 255, (*self.image_cfg.input_size[1:],
                                           self.image_cfg.input_size[0])).to(
                                               torch.float32)

        image = self.transforms(image=np.array(image))["image"]
        return {"total_calories": total_calories, "image": image, "ingredients": ingredients}


def collate_fn(batch, tokenizer):
    ingredients = [item["ingredients"] for item in batch]
    images = torch.stack([item["image"] for item in batch])
    total_calories = torch.LongTensor([item["total_calories"] for item in batch])

    tokenized_input = tokenizer(ingredients,
                                return_tensors="pt",
                                padding="max_length",
                                truncation=True)

    return {
        "total_calories": total_calories,
        "image": images,
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"]
    }


def get_transforms(config, ds_type="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

    if ds_type == "train":
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.ColorJitter(brightness=0.2,
                              contrast=0.2,
                              saturation=0.2,
                              p=0.7),
                A.Rotate(limit=[-90,90],
                         p=0.7
                ),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=seed,
        )
    else:
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.CenterCrop(
                    height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=seed,
        )

    return transforms
