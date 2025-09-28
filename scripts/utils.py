import os
import random
from functools import partial

import numpy as np

import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import torchmetrics

from transformers import AutoModel, AutoTokenizer

from scripts.dataset import MultimodalDataset, collate_fn, get_transforms



def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def set_requires_grad(module: nn.Module, unfreeze_pattern="", verbose=False):
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = True # размораживаем все слои
        return

    pattern = unfreeze_pattern.split("|")

    for name, param in module.named_parameters():
        if any([name.startswith(p) for p in pattern]):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False


class MultimodalRegressionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.HIDDEN_DIM)
        self.image_proj = nn.Linear(self.image_model.num_features, config.HIDDEN_DIM)

        # Слой для регрессии - с 1 выходом
        self.regressor = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.LayerNorm(config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(config.HIDDEN_DIM // 2, 1)
        )

    def forward(self, input_ids, attention_mask, image):
        text_features = self.text_model(input_ids, attention_mask).last_hidden_state[:, 0, :]
        image_features = self.image_model(image)

        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)

        fused_emb = text_emb * image_emb

        prediction = self.regressor(fused_emb)
        return prediction.squeeze(-1)


def train(config, device):
    seed_everything(config.SEED)

    # Инициализация модели
    model = MultimodalRegressionModel(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    set_requires_grad(model.text_model,
                      unfreeze_pattern=config.TEXT_MODEL_UNFREEZE, verbose=True)
    set_requires_grad(model.image_model,
                      unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE, verbose=True)

    # Оптимизатор с разными LR
    optimizer = AdamW([{
        'params': model.text_model.parameters(),
        'lr': config.TEXT_LR
    }, {
        'params': model.image_model.parameters(),
        'lr': config.IMAGE_LR
    }, {
        'params': model.regressor.parameters(),
        'lr': config.REGRESSOR_LR
    }])

    criterion = nn.L1Loss()  # nn.L1Loss() - MAE

    # Загрузка данных
    transforms = get_transforms(config)
    val_transforms = get_transforms(config, ds_type="val")
    train_dataset = MultimodalDataset(config, transforms)
    val_dataset = MultimodalDataset(config, val_transforms, ds_type="val")
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              collate_fn=partial(collate_fn,
                                                 tokenizer=tokenizer))
    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            collate_fn=partial(collate_fn,
                                               tokenizer=tokenizer))

    mae_metric_train = torchmetrics.MeanAbsoluteError().to(device)
    mae_metric_val = torchmetrics.MeanAbsoluteError().to(device)
    
    best_val_loss = float('inf')

    print("Training started")
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            # Подготовка данных
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device)
            }
            targets = batch['total_calories'].float().to(device)

            # Forward
            optimizer.zero_grad()
            predictions = model(**inputs)
            loss = criterion(predictions, targets)

            # Backward
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Метрики
            _ = mae_metric_train(predictions, targets)

        # Валидация
        train_mae = mae_metric_train.compute().cpu().numpy()
        val_mae = validate_regression(model, val_loader, device, mae_metric_val)
        
        mae_metric_val.reset()
        mae_metric_train.reset()

        print(
            f"Epoch {epoch}/{config.EPOCHS-1} | "
            f"Avg L1 Loss: {total_loss/len(train_loader):.4f} | "
            f"Train MAE: {train_mae:.4f} | "
            f"Val MAE: {val_mae:.4f}"
        )

        if val_mae < best_val_loss:
            print(f"New best model, epoch: {epoch}")
            best_val_loss = val_mae
            torch.save(model.state_dict(), config.SAVE_PATH)


def validate_regression(model, val_loader, device, mae_metric):
    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device)
            }
            targets = batch['total_calories'].float().to(device)

            predictions = model(**inputs)
            
            _ = mae_metric(predictions, targets)

    return mae_metric.compute().cpu().numpy()
