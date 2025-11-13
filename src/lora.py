import torch

from functools import partial
from timm.models.vision_transformer import VisionTransformer


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std = torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) / std)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.scaler = alpha / rank

    def forward(self, x):
        x = self.scaler * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def create_lora_model(model: VisionTransformer, lora_rank: int, lora_alpha: int) -> VisionTransformer:
    assign_lora = partial(LinearWithLoRA, rank=lora_rank, alpha=lora_alpha)
    for block in model.blocks:
        block.attn.qkv = assign_lora(block.attn.qkv)
        block.attn.proj = assign_lora(block.attn.proj)
        block.mlp.fc1 = assign_lora(block.mlp.fc1)
        block.mlp.fc2 = assign_lora(block.mlp.fc2)

    return model
