"""
Configuration management for Aegis-JKN ML/DL Backend
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional

ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"

class Settings(BaseSettings):
    """Application settings"""

    # Notion API
    notion_token: str = ""
    notion_page_id: str = "3f23e603-1ab7-808d-b2cb-fe92cd62391d"
    notion_dataset_db_id: str = "310b35a5-9346-42df-af47-df9897cdcda1"
    notion_claims_db_id: str = "c783afb3-d0d4-4582-9545-6180b7b7756b"

    # OpenAI API
    openai_api_base: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Database
    database_url: str = "postgresql://user:password@localhost:5432/aegis_jkn_ml"
    redis_url: str = "redis://localhost:6379/0"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    # ML/DL Configuration
    gcn_hidden_dim: int = 128
    gcn_num_layers: int = 3
    attention_heads: int = 4
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100

    # OASIS Configuration
    oasis_num_agents: int = 100
    oasis_simulation_steps: int = 10
    oasis_activation_probability: float = 0.1

    class Config:
        env_file = str(ENV_PATH) if ENV_PATH.exists() else ".env"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()
