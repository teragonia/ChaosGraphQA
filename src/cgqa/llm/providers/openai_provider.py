"""OpenAI LLM provider implementation."""

import os
from typing import Any, Dict, List, Optional

from .base import BaseLLMProvider, LLMConfig, LLMResponse

try:
    import openai
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider using the official OpenAI API."""

    SUPPORTED_MODELS = [
        "gpt-5",
        "gpt-4o",
        "gpt-4o-2024-11-20",
        "gpt-4.1-2025-04-14",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "o1-preview",
        "o1-mini",
    ]

    def __init__(self, config: LLMConfig):
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )

        if not config.api_key:
            config.api_key = os.getenv("OPENAI_API_KEY")

        if not config.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key in config."
            )

        super().__init__(config)

    def _setup_client(self) -> None:
        """Initialize the OpenAI client."""
        client_kwargs = {
            "api_key": self.config.api_key,
            "timeout": self.config.timeout,
        }

        if self.config.api_base:
            client_kwargs["base_url"] = self.config.api_base

        self.client = OpenAI(**client_kwargs)

    def _make_request(self, prompt: str, **kwargs) -> LLMResponse:
        """Make a request to OpenAI API using the responses interface."""

        is_reasoning_model = self.config.model_name.startswith(
            "o1"
        ) or self.config.model_name.startswith("gpt-5")
        request_params = {
            "model": self.config.model_name,
            "input": [{"role": "user", "content": prompt}],
        }

        if not is_reasoning_model:
            request_params["temperature"] = kwargs.get(
                "temperature", self.config.temperature
            )
        for key, value in kwargs.items():
            if key not in ["max_tokens", "temperature"] and not key.startswith("_"):
                if not is_reasoning_model or key not in [
                    "temperature",
                    "top_p",
                    "frequency_penalty",
                    "presence_penalty",
                ]:
                    request_params[key] = value

        try:
            response = self.client.responses.create(**request_params)
            response_text = response.output_text or ""

            return LLMResponse(
                text=response_text,
                model_name=self.config.model_name,
                provider_name="openai",
                request_id=getattr(response, "id", None),
                prompt_tokens=None,  # responses API may not provide detailed token counts
                completion_tokens=None,
                total_tokens=None,
                raw_response=(
                    response.model_dump() if hasattr(response, "model_dump") else None
                ),
            )

        except Exception as e:
            return LLMResponse(
                text="",
                model_name=self.config.model_name,
                provider_name="openai",
                error=str(e),
                error_type=type(e).__name__,
            )

    def test_connection(self) -> bool:
        """Test connection to OpenAI API."""
        try:
            response = self.generate("Hello", max_tokens=1, temperature=0.1)
            return response.error is None
        except Exception:
            return False

    def validate_config(self) -> List[str]:
        """Validate OpenAI-specific configuration."""
        errors = super().validate_config()

        if not self.config.api_key:
            errors.append("api_key is required for OpenAI provider")

        if self.config.model_name not in self.SUPPORTED_MODELS:
            errors.append(
                f"Unsupported model '{self.config.model_name}'. "
                f"Supported models: {', '.join(self.SUPPORTED_MODELS)}"
            )

        return errors

    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information."""
        info = super().get_model_info()
        is_reasoning_model = self.config.model_name.startswith(
            "o1"
        ) or self.config.model_name.startswith("gpt-5")
        info.update(
            {
                "provider_specific": {
                    "api_version": "v1",
                    "supports_functions": not is_reasoning_model,
                    "supports_system_messages": not is_reasoning_model,
                    "supports_temperature": not is_reasoning_model,
                    "is_reasoning_model": is_reasoning_model,
                    "context_window": self._get_context_window(),
                    "training_data_cutoff": self._get_training_cutoff(),
                }
            }
        )
        return info

    def _get_context_window(self) -> int:
        """Get context window size for the model."""
        return {
            "gpt-5": 400000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "o1-preview": 128000,
            "o1-mini": 128000,
        }.get(self.config.model_name, 4096)

    def _get_training_cutoff(self) -> str:
        """Get training data cutoff for the model."""
        return {
            "gpt-5": "September 30, 2024",
            "gpt-4o": "October 2023",
            "gpt-4o-mini": "October 2023",
            "gpt-4-turbo": "April 2024",
            "gpt-4": "September 2021",
            "gpt-3.5-turbo": "September 2021",
            "gpt-3.5-turbo-16k": "September 2021",
            "o1-preview": "October 2023",
            "o1-mini": "October 2023",
        }.get(self.config.model_name, "Unknown")

    @classmethod
    def create_config(
        cls,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        **kwargs,
    ) -> LLMConfig:
        """Create a configuration for OpenAI provider.

        Args:
            model: OpenAI model name
            api_key: API key (uses OPENAI_API_KEY env var if not provided)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional configuration parameters

        Returns:
            LLMConfig object
        """
        return LLMConfig(
            provider_name="openai",
            model_name=model,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
