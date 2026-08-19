"""Load and validate environment variables"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_SERVICE_ROLE_KEY: str
    WEBHOOK_SECRET: str
    BUCKET_NAME: str = "photos"
    TARGET_WIDTH: int = 480
    TARGET_HEIGHT: int = 800
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
@lru_cache
def get_settings() -> Settings:
    return Settings()