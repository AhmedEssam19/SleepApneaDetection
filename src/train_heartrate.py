import torch
import random
import typer
import wandb

import numpy as np
import lightning as L

from torch.backends import cudnn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from torchaudio import transforms as audio_transforms
from torchvision import transforms
from typing import Literal, Annotated
from lightning.pytorch import callbacks
from lightning.pytorch.loggers import WandbLogger
from dataset import HeartRateDataset
from model import HeartRateModel
from config import CONFIG


torch.set_float32_matmul_precision('high')

app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
    data_path: Annotated[str, typer.Option()],
    vit_size: Annotated[Literal["small", "medium", "large"], typer.Option()],
    finetuning_method: Annotated[Literal["scratch", "head", "full", "lora"], typer.Option()],
    use_augmentation: bool = False,
    patch_size: int = 16,
    in_chans: int = 1,
    rank: int = 4,
    alpha: float = 16,
    pretrained_vit_path: str = None,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    warmup_steps: int = 500,
    learning_rate_patience: int = 2,
    num_workers: int = 4,
    early_stopping_patience: int = 5,
    logging_steps: int = 1,
    seed: int = 42
):
    # fix the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    npz = np.load(data_path)
    X, y = npz['X'], npz['y']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed
    )
    
    transform = transforms.Compose([
        transforms.Lambda(lambda x: 10 * torch.log10(x + 1e-12)),
        transforms.Resize((224, 224), antialias=True, interpolation=transforms.InterpolationMode.BICUBIC),
    ])
    dataset = HeartRateDataset(X_train, y_train, transform=transform)
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
    
    train_dataset = HeartRateDataset(X_train, y_train, transform=transform_train)
    val_dataset = HeartRateDataset(X_val, y_val, transform=transform_validation)
    test_dataset = HeartRateDataset(X_test, y_test, transform=transform_validation)
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

    checkpoint_filename = wandb_logger.experiment.name + "-{epoch}-{step}-{val_loss:.4f}"
    train_callbacks = [
        callbacks.ModelCheckpoint(monitor="val_loss", mode="min", dirpath=CONFIG.checkpoint_dir, filename=checkpoint_filename),
        callbacks.EarlyStopping(monitor="val_loss", patience=early_stopping_patience, mode="min"),
        callbacks.LearningRateMonitor(logging_interval='step'),
        callbacks.RichProgressBar(),
    ]

    trainer = L.Trainer(
        max_epochs=-1,
        callbacks=train_callbacks,
        logger=wandb_logger,
        log_every_n_steps=logging_steps,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )

    model = HeartRateModel(
        vit_size=vit_size,
        finetuning_method=finetuning_method,
        in_chans=in_chans,
        patch_size=patch_size,
        num_classes=1,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        learning_rate_patience=learning_rate_patience,
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


if __name__ == "__main__":
    app()
