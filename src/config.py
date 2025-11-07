import yaml


from pydantic_settings import BaseSettings


class SpectrogramConfig(BaseSettings):
    sample_rate: int
    window_size: int
    window_overlap: int


class Config(BaseSettings):
    spectrogram: SpectrogramConfig


with open("config.yml", "r") as f:
    config_dict = yaml.safe_load(f)

CONFIG = Config(**config_dict)
