import torch
import random
import typer
import wandb
import glob
import os

import numpy as np
import lightning as L

from torch.backends import cudnn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchaudio import transforms as audio_transforms
from torchvision import transforms
from sklearn.model_selection import train_test_split
from typing import Literal, Annotated
from lightning.pytorch import callbacks
from lightning.pytorch.loggers import WandbLogger
from dataset import SleepApneaDataset
from model import EEGModel
from config import CONFIG


torch.set_float32_matmul_precision('high')

app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
    data_path: Annotated[str, typer.Option()],
    vit_size: Annotated[Literal["small", "medium", "large"], typer.Option()],
    finetuning_method: Annotated[Literal["scratch", "head", "full", "lora"], typer.Option()],
    patch_size: int = 16,
    rank: int = 4,
    alpha: float = 16,
    pretrained_vit_path: str = None,
    learning_rate: float = 1e-4,
    use_augmentation: bool = False,
    batch_size: int = 32,
    num_workers: int = 4,
    early_stopping_patience: int = 10,
    logging_steps: int = 1,
    seed: int = 42
):
    # fix the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    data = []
    folds = []
    for file_path in glob.glob(f"{data_path}/signals/**/*.npy", recursive=True):
        loaded_data = np.load(file_path)
        data.append(loaded_data)
        fold_name = os.path.basename(os.path.dirname(file_path))
        folds.extend([int(fold_name[-1])] * loaded_data.shape[0])

    data = np.concatenate(data, axis=0)
    folds = np.array(folds)

    labels = []
    for file_path in glob.glob(f"{data_path}/labels/**/*.npy", recursive=True):
        loaded_labels = np.load(file_path)
        labels.append(loaded_labels)
    labels = np.concatenate(labels, axis=0)
    labels = labels.astype(np.float32)

    for test_fold in range(9):
        X_test = data[folds == test_fold]
        y_test = labels[folds == test_fold]
        val_fold = (test_fold + 1) % 9
        X_val = data[folds == val_fold]
        y_val = labels[folds == val_fold]
        X_train = data[(folds != test_fold) & (folds != val_fold)]
        y_train = labels[(folds != test_fold) & (folds != val_fold)]

        train_fold(
            X_train, y_train, X_val, y_val, X_test, y_test,
            vit_size=vit_size,
            finetuning_method=finetuning_method,
            patch_size=patch_size,
            rank=rank,
            alpha=alpha,
            pretrained_vit_path=pretrained_vit_path,
            learning_rate=learning_rate,
            use_augmentation=use_augmentation,
            batch_size=batch_size,
            num_workers=num_workers,
            early_stopping_patience=early_stopping_patience,
            logging_steps=logging_steps,
        )


def train_fold(
    X_train, 
    y_train,
    X_val, 
    y_val,
    X_test,
    y_test,
    vit_size: Literal["small", "medium", "large"],
    finetuning_method: Literal["scratch", "head", "full", "lora"],
    patch_size: int,
    rank: int,
    alpha: float,
    pretrained_vit_path: str,
    learning_rate: float,
    use_augmentation: bool,
    batch_size: int,
    num_workers: int,
    early_stopping_patience: int,
    logging_steps: int,
):
        
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
            transforms.Resize((224, 224), antialias=True,
                            interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize(mean=mean, std=std),
            audio_transforms.TimeMasking(time_mask_param=50),
            audio_transforms.FrequencyMasking(freq_mask_param=50),
        ])
    else:
        transform_train = transform_validation

    train_dataset = SleepApneaDataset(X_train, y_train, transform=transform_train)
    val_dataset = SleepApneaDataset(X_val, y_val, transform=transform_validation)
    test_dataset = SleepApneaDataset(X_test, y_test, transform=transform_validation)
    sampler_train = RandomSampler(train_dataset)
    sampler_val = SequentialSampler(val_dataset)
    sampler_test = SequentialSampler(test_dataset)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler_train, num_workers=num_workers, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=sampler_val, num_workers=num_workers, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, sampler=sampler_test, num_workers=num_workers, pin_memory=True
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

    model = EEGModel(
        vit_size=vit_size,
        finetuning_method=finetuning_method,
        patch_size=patch_size,
        num_classes=1,
        learning_rate=learning_rate,
        rank=rank,
        alpha=alpha,
        pretrained_vit_path=pretrained_vit_path
    )
    wandb_logger.watch(model, log="gradients", log_freq=10)
    wandb_logger.log_hyperparams({
        "batch_size": batch_size,
        "early_stopping_patience": early_stopping_patience,
        "use_augmentation": use_augmentation
    })

    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(dataloaders=test_dataloader, ckpt_path="best")
    wandb.finish()


if __name__ == "__main__":
    app()
