from pathlib import Path
from typing import List
from functools import lru_cache

import yaml
from pydantic import BaseModel


class DeepSeekConfig(BaseModel):
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-v4-pro"
    timeout: int = 60
    max_retries: int = 3


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = ["*"]


class AppConfig(BaseModel):
    deepseek: DeepSeekConfig = DeepSeekConfig()
    server: ServerConfig = ServerConfig()


@lru_cache
def get_config() -> AppConfig:
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}
        return AppConfig(**data)
    return AppConfig()
