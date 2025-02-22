import litellm
from config import Config

class LiteLLMClient:
    def __init__(self):
        """Initialize LiteLLM client with model configurations."""
        self.model_config = Config.MODEL_CONFIG

    def list_available_models(self):
        """Returns a list of available models."""
        return list(self.model_config.keys())

    def generate_response(self, model: str, messages: list, system_prompt: str = None) -> str:
        if model not in self.model_config:
            raise ValueError(f"Model '{model}' not found. Available models: {self.list_available_models()}")

        # Add system prompt if provided
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        response = litellm.completion(
            model=model,
            messages=messages,
            model_config=self.model_config
        )

        return response["choices"][0]["message"]["content"]