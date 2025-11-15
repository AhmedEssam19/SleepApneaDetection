import torch
import random
import typer
import wandb

import numpy as np
import pandas as pd
import lightning as L

from torch.backends import cudnn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from typing import Literal, Annotated
from lightning.pytorch import callbacks
from lightning.pytorch.loggers import WandbLogger
from dataset import SleepApneaDataset
from model import SleepApneaModel
from config import CONFIG

app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
        data_path: Annotated[str, typer.Option()],
        labels_path: Annotated[str, typer.Option()],
        vit_size: Annotated[Literal["small", "medium", "large"], typer.Option()],
        finetuning_method: Annotated[Literal["scratch", "head", "full", "lora"], typer.Option()],
        rank: int = 4,
        alpha: float = 16,
        pretrained_vit_path: str = None,
        learning_rate: float = 1e-4,
        label_smoothing: float = 0.0,
        use_augmentation: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
        early_stopping_patience: int = 20,
        logging_steps: int = 1,
        seed: int = 42
    ):
    # fix the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    data = pd.read_csv(data_path)
    data = np.array(np.split(data.values, data.shape[0] // 170))
    data = data.transpose((0, 2, 1))
    labels = pd.read_csv(labels_path).to_numpy().squeeze()

    X_train, X_val, y_train, y_val = train_test_split(
        data, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    transform = transforms.Compose([
        transforms.Lambda(lambda x: 10 * torch.log10(x + 1e-12)),
        transforms.Resize((224, 224), antialias=True, interpolation=transforms.InterpolationMode.BICUBIC),
    ])
    dataset = SleepApneaDataset(X_train, y_train, transform=transform)
    mean, std = dataset.get_statistics()

    transform_validation = transforms.Compose([
        transforms.Lambda(lambda x: 10 * torch.log10(x + 1e-12)),
        transforms.Resize((224, 224), antialias=True,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Normalize(mean=mean, std=std)
    ])

    if use_augmentation:
        transform_train = transforms.Compose([
            transforms.Lambda(lambda x: 10 * torch.log10(x + 1e-12)),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((14, 7)),
            transforms.Resize((224, 224), antialias=True,
                            interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        transform_train = transform_validation

    train_dataset = SleepApneaDataset(X_train, y_train, transform=transform_train)
    val_dataset = SleepApneaDataset(X_val, y_val, transform=transform_validation)
    sampler_train = RandomSampler(train_dataset)
    sampler_val = SequentialSampler(val_dataset)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler_train, num_workers=num_workers, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=sampler_val, num_workers=num_workers, pin_memory=True
    )

    wandb.login(key=CONFIG.wandb.key)
    wandb_logger = WandbLogger(project=CONFIG.wandb.project_name)

    checkpoint_filename = wandb_logger.experiment.name + "-{epoch}-{step}-{val_acc:.4f}"
    train_callbacks = [
        callbacks.ModelCheckpoint(monitor="val_acc", mode="max", dirpath=CONFIG.checkpoint_dir, filename=checkpoint_filename),
        callbacks.EarlyStopping(monitor="val_acc", patience=early_stopping_patience, mode="max"),
        callbacks.LearningRateMonitor(logging_interval='epoch'),
        callbacks.RichProgressBar(),
    ]

    trainer = L.Trainer(
        max_epochs=-1,
        callbacks=train_callbacks,
        logger=wandb_logger,
        log_every_n_steps=logging_steps,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )

    model = SleepApneaModel(
        vit_size=vit_size,
        finetuneing_method=finetuning_method,
        num_classes=len(np.unique(labels)),
        learning_rate=learning_rate,
        label_smoothing=label_smoothing,
        rank=rank,
        alpha=alpha,
        pretrained_vit_path=pretrained_vit_path
    )
    wandb_logger.watch(model, log="gradients", log_freq=10)
    wandb_logger.log_hyperparams({
        "batch_size": batch_size,
        "early_stopping_patience": early_stopping_patience,
        "label_smoothing": label_smoothing,
        "use_augmentation": use_augmentation
    })

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    app()
