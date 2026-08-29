from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DATA_ROOT: str = "/data/nuscenes"
    APP_CONFIG_PATH: str = "/app/config/settings.yml"

    POSTGRES_HOST:     str = "db"
    POSTGRES_PORT:     str = "5432"
    POSTGRES_DB:       str
    POSTGRES_USER:     str
    POSTGRES_PASSWORD: str
    @model_validator(mode="after")
    def build_database_url(self) -> "Settings":
        ssl_suffix = "?ssl=require" if self.DEPLOY_ENV == "aws" else ""
        self.DATABASE_URL = (
            f"postgresql+asyncpg://"
            f"{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}"
            f"/{self.POSTGRES_DB}"
            f"{ssl_suffix}"
        )
        return self

    DATABASE_URL: str = ""

settings = Settings()
