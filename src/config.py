import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    """Configuration class to manage API keys."""
    MODEL_CONFIG = {
        "gpt-4": {"api_key": os.getenv("OPENAI_API_KEY")},
        "claude-3-sonnet": {"api_key": os.getenv("ANTHROPIC_API_KEY")},
        "deepseek-chat": {"api_key": os.getenv("DEEPSEEK_API_KEY")},
    }