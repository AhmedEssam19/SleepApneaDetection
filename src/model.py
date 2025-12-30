import torch
import timm.models.vision_transformer
import torchmetrics

import torch.nn as nn
import lightning as L

from functools import partial
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import AdamW
from timm.models._manipulate import checkpoint_seq
from typing import Literal
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lora import create_lora_model


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool, tanh=False, head_layers=1, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.global_pool = global_pool
        self.tanh = tanh
        num_classes = kwargs['num_classes']
        layers = []
        for i in range(head_layers - 1):
            layers.extend([nn.Linear(self.embed_dim, self.embed_dim), nn.ReLU()])
        layers.append(nn.Linear(self.embed_dim, num_classes))
        self.head = nn.Sequential(*layers) if head_layers > 1 else nn.Linear(self.embed_dim, num_classes)

    def unfreeze_patch_embed(self):
        for param in self.patch_embed.parameters():
            param.requires_grad = True

    def freeze_encoder(self, num_blocks=None):
        if num_blocks is None:
            for param in self.blocks.parameters():
                param.requires_grad = False
        else:
            for param in self.blocks[:num_blocks].parameters():
                param.requires_grad = False

    def freeze_encoder_lora(self):
        # Freeze all params
        for param in self.blocks.parameters():
            param.requires_grad = False

        # Unfreeze LoRA layers
        for block in self.blocks:
            for param in block.attn.qkv.lora.parameters():
                param.requires_grad = True

            for param in block.attn.proj.lora.parameters():
                param.requires_grad = True

            for param in block.mlp.fc1.lora.parameters():
                param.requires_grad = True

            for param in block.mlp.fc2.lora.parameters():
                param.requires_grad = True


    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        if self.tanh:
            return torch.tanh(x)
        return x


def vit_small_patch16(**kwargs):
    model = VisionTransformer(embed_dim=512, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_medium_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class PLModel(L.LightningModule):
    def _setup_finetuning(self, finetuning_method: Literal["scratch", "head", "full", "lora"], rank: int, alpha: int, checkpoint_path: str = None):
        if finetuning_method != "scratch":
            if checkpoint_path is None:
                raise ValueError("Checkpoint path must be provided for finetuning.")
            
            self._load_pretrained_weights(checkpoint_path)
        
        if finetuning_method == "head":
            self.vit.freeze_encoder()
        elif finetuning_method == "lora":
            self.vit = create_lora_model(self.vit, rank, alpha)
            self.vit.freeze_encoder_lora()
        elif finetuning_method != "scratch" and finetuning_method != "full":
            raise ValueError(f"Unknown finetuning_method: {finetuning_method}")
            
    def _load_pretrained_weights(self, checkpoint_path: str):    
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        for key in list(checkpoint['model'].keys()):
            if key.startswith('head.') or key.startswith('patch_embed.'):
                print("Removing key from pretrained weights:", key)
                del checkpoint['model'][key]
        incompatible_keys = self.vit.load_state_dict(checkpoint['model'], strict=False)
        print("Missing keys when loading pretrained weights:", incompatible_keys.missing_keys)

    def forward(self, inputs):
        output = self.vit(inputs)
        return output

    def training_step(self, batch, _):
        inputs, labels = batch
        preds = self(inputs)
        loss = self.loss_fn(preds, labels)
        self.train_acc(preds, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log(
            "train_acc",
            self.train_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, _):
        inputs, labels = batch
        preds = self(inputs)
        loss = self.loss_fn(preds, labels)
        self.val_acc(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "val_acc",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        optimizer_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optimizer_scheduler,
                "monitor": "val_loss"
            }
        }


class SleepApneaModel(PLModel):
    def __init__(
        self, 
        vit_size: Literal["small", "medium", "large"],
        finetuning_method: Literal["scratch", "head", "full", "lora"],
        patch_size: int,
        num_classes: int,
        learning_rate: float,
        label_smoothing: float,
        rank: int,
        alpha: float,
        pretrained_vit_path: str = None
    ):
        super().__init__()
        self.vit = self._init_vit(vit_size, num_classes, patch_size)
        self._setup_finetuning(finetuning_method, rank, alpha, pretrained_vit_path)
        self.loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def _init_vit(self, vit_size: Literal["small", "medium", "large"], num_classes: int, patch_size: int):
        vit = {
            "small": vit_small_patch16,
            "medium": vit_medium_patch16,
            "large": vit_large_patch16
        }
        return vit[vit_size](num_classes=num_classes, global_pool="token", in_chans=5, patch_size=patch_size)
    


class EEGModel(PLModel):
    def __init__(self, 
        vit_size: Literal["small", "medium", "large"],
        finetuning_method: Literal["scratch", "head", "full", "lora"],
        patch_size: int,
        num_classes: int,
        learning_rate: float,
        rank: int,
        alpha: float,
        pretrained_vit_path: str = None
    ):
        super().__init__()
        self.vit = self._init_vit(vit_size, num_classes, patch_size)
        self._setup_finetuning(finetuning_method, rank, alpha, pretrained_vit_path)
        self.loss_fn = BCEWithLogitsLoss()
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def _init_vit(self, vit_size: Literal["small", "medium", "large"], num_classes: int, patch_size: int):
        vit = {
            "small": vit_small_patch16,
            "medium": vit_medium_patch16,
            "large": vit_large_patch16
        }
        return vit[vit_size](num_classes=num_classes, global_pool="token", in_chans=3, patch_size=patch_size)
    
    def forward(self, inputs):
        output = self.vit(inputs).view(-1)
        return output
    
    def test_step(self, batch, _):
        inputs, labels = batch
        preds = self(inputs)
        loss = self.loss_fn(preds, labels)
        self.test_acc(preds, labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "test_acc",
            self.test_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
