"""
Configuration module for Sue AI Assistant with enhanced API key checks
"""
import os
from pathlib import Path
from typing import List
from dataclasses import dataclass
from dotenv import load_dotenv, set_key

# Определяем путь к файлу .env
dotenv_path = Path('.env')

# Загружаем переменные окружения из .env файла
load_dotenv(dotenv_path=dotenv_path)

@dataclass
class AppConfig:
    """Application configuration"""
    # Model settings
    MODEL_NAME: str = "mistral-large-latest"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 1000

    # API Keys
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")

    # Cache settings
    CACHE_TTL: int = 3600  # 1 hour
    WEB_CACHE_SIZE: int = 100

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Web scraping
    REQUEST_TIMEOUT: int = 30
    USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

    def validate(self) -> List[str]:
        """Validate configuration"""
        issues = []

        if not self.MISTRAL_API_KEY:
            issues.append("MISTRAL_API_KEY is not set")

        return issues

    def ensure_api_keys(self):
        """Ensure that all required API keys are set in .env"""
        # Reload environment variables to get the latest values
        load_dotenv(dotenv_path=dotenv_path, override=True)
        
        # Check if MISTRAL_API_KEY exists in environment after reload
        self.MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
        
        if not self.MISTRAL_API_KEY:
            print("API ключ Mistral отсутствует.")
            self.request_mistral_key()

    def request_mistral_key(self):
        """Request Mistral API key if not set and save it to .env"""
        self.MISTRAL_API_KEY = input("Введите ваш Mistral API ключ: ").strip()
        if not self.MISTRAL_API_KEY:
            raise ValueError("API ключ не может быть пустым.")

        # Сохраняем ключ в .env, обновляя или добавляя переменную
        try:
            set_key(dotenv_path, "MISTRAL_API_KEY", self.MISTRAL_API_KEY)
            print("Mistral API ключ успешно сохранен.")
        except Exception as e:
            raise RuntimeError(f"Не удалось сохранить ключ в .env: {e}")

        # Перезагружаем переменные окружения после сохранения
        load_dotenv(dotenv_path=dotenv_path, override=True)

# Create global config instance
config = AppConfig()

# Ensure API keys are present
config.ensure_api_keys()

# Validate configuration
issues = config.validate()
if issues:
    for issue in issues:
        print(f"[Проблема конфигурации]: {issue}")
    raise RuntimeError("Конфигурация приложения имеет ошибки. Исправьте их и перезапустите.")
