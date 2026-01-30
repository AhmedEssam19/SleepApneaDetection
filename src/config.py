import yaml

from pydantic_settings import BaseSettings
from pydantic import Field


class SpectrogramConfig(BaseSettings):
    sample_rate: int
    window_size: int
    window_overlap: int
    max_freq: float
    min_freq: float

class WandBConfig(BaseSettings):
    project_name: str
    key: str = Field(validation_alias="WANDB_KEY")

class Config(BaseSettings):
    spectrogram: SpectrogramConfig
    wandb: WandBConfig
    checkpoint_dir: str


with open("config.yml", "r") as f:
    config_dict = yaml.safe_load(f)

CONFIG = Config(**config_dict)
