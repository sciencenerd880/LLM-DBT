import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    """Configuration class to manage API keys."""
    MODEL_CONFIG = {
        # OPENAI API KEY
        "gpt-3.5-turbo": {"api_key": os.getenv("OPENAI_API_KEY")},
        "gpt-4": {"api_key": os.getenv("OPENAI_API_KEY")},
        "o1-mini": {"api_key": os.getenv("OPENAI_API_KEY")},
        "o1-preview": {"api_key": os.getenv("OPENAI_API_KEY")},

        "deepseek-chat": {"api_key": os.getenv("DEEPSEEK_API_KEY")},
        
    }