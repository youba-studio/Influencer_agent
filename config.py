"""
Centralized Configuration Loader for Anaya Sharma AI Influencer Agent.
Reads all credentials strictly from the local .env file. Never hardcodes secrets.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Path definitions
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    """Immutable secure configuration values loaded from .env."""
    google_api_key: str
    hf_token: str
    instagram_user: str
    instagram_pass: str
    kaggle_username: str
    kaggle_key: str
    kaggle_api_token: str
    colab_api_key: str
    colab_webhook_url: str
    earnkaro_api_key: str
    earnkaro_user: str
    earnkaro_pass: str
    linktree_account_id: str

    @classmethod
    def load(cls) -> "AppConfig":
        return cls(
            google_api_key=os.getenv("GOOGLE_API_KEY", "").strip(),
            hf_token=os.getenv("HF_TOKEN", "").strip(),
            instagram_user=os.getenv("INSTAGRAM_USER", "").strip(),
            instagram_pass=os.getenv("INSTAGRAM_PASS", "").strip(),
            kaggle_username=os.getenv("KAGGLE_USERNAME", "").strip(),
            kaggle_key=os.getenv("KAGGLE_KEY", "").strip(),
            kaggle_api_token=os.getenv("KAGGLE_API_TOKEN", os.getenv("KAGGLE_KEY", "")).strip(),
            colab_api_key=os.getenv("COLAB_API_KEY", "").strip(),
            colab_webhook_url=os.getenv("COLAB_WEBHOOK_URL", "").strip(),
            earnkaro_api_key=os.getenv("EARNKARO_API_KEY", "").strip(),
            earnkaro_user=os.getenv("EARNKARO_USER", "").strip(),
            earnkaro_pass=os.getenv("EARNKARO_PASS", "").strip(),
            linktree_account_id=os.getenv("LINKTREE_ACCOUNT_ID", "anaya_sharma_style").strip(),
        )

    def validate(self) -> dict[str, bool]:
        """Validates credential presence across all pipeline integrations."""
        return {
            "huggingface": bool(self.hf_token),
            "kaggle": bool(self.kaggle_key or self.kaggle_api_token),
            "instagram": bool(self.instagram_user and self.instagram_pass),
            "earnkaro": bool(self.earnkaro_api_key or (self.earnkaro_user and self.earnkaro_pass)),
            "colab": bool(self.colab_api_key or self.colab_webhook_url),
            "gemini": bool(self.google_api_key),
        }


# Singleton config
config = AppConfig.load()
