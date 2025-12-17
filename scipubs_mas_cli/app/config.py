import os
from dataclasses import dataclass


@dataclass
class Settings:
    """Application and infrastructure configuration.

    LLM endpoint is assumed to be OpenAI-compatible (e.g., LiteLLM proxy).
    PostgreSQL settings are aligned with docker-compose defaults.
    """

    litellm_base_url: str = os.getenv("LITELLM_BASE_URL", "http://a6k2.dgx:34000/v1")
    # litellm_api_key: str = os.getenv("LITELLM_API_KEY", "sk-1yvtuYMQN37uRpXQe44qrA")
    litellm_api_key: str = os.getenv("LITELLM_API_KEY", "sk-wJOTEk68IfYmW5ePxtdpYQ") # Rosetta
    model_name: str = os.getenv("MODEL_NAME", "qwen3-32b")

    postgres_host: str = os.getenv("POSTGRES_HOST", "176.32.34.156")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "postgres")
    postgres_user: str = os.getenv("POSTGRES_USER", "user")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "pass")

    # OpenAlex is optional, but providing a mailto is recommended by OpenAlex.
    openalex_mailto: str | None = os.getenv("OPENALEX_MAILTO")

    # NOTE: Collector uses SQL by default and will call OpenAlex only when the user
    # explicitly requests it for the current query. The variable below is kept only
    # for backwards compatibility and is not used as a default "auto" fallback.
    collector_source: str = os.getenv("COLLECTOR_SOURCE", "sql")


settings = Settings()

# Explicit constants for agents (as required by the task)
BASE_URL = settings.litellm_base_url
API_KEY = settings.litellm_api_key
MODEL_NAME = settings.model_name
