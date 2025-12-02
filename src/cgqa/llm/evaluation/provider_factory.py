"""Factory for creating LLM providers."""

import os
from typing import Any, Dict, Optional, Type, Union

from ..providers.anthropic_provider import AnthropicProvider
from ..providers.base import BaseLLMProvider, LLMConfig
from ..providers.gemini_provider import GeminiProvider
from ..providers.huggingface_provider import HuggingFaceProvider
from ..providers.openai_provider import OpenAIProvider


class ProviderFactory:
    """Factory class for creating LLM providers."""

    PROVIDERS: Dict[str, Type[BaseLLMProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "huggingface": HuggingFaceProvider,
    }

    MODEL_TO_PROVIDER = {
        "gpt-5": "openai",
        "gpt-4o": "openai",
        "gpt-4o-mini": "openai",
        "gpt-4-turbo": "openai",
        "gpt-4": "openai",
        "gpt-3.5-turbo": "openai",
        "o1-preview": "openai",
        "o1-mini": "openai",
        "claude-3.5-sonnet": "anthropic",
        "claude-3.5-haiku": "anthropic",
        "claude-3-opus": "anthropic",
        "claude-3-sonnet": "anthropic",
        "claude-3-haiku": "anthropic",
        "gemini-1.5-pro": "gemini",
        "gemini-1.5-flash": "gemini",
        "gemini-1.0-pro": "gemini",
        "gemini-exp-1114": "gemini",
        "gemini-exp-1121": "gemini",
    }

    @classmethod
    def create_provider(
        self,
        model_identifier: str,
        config: Optional[Union[LLMConfig, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> BaseLLMProvider:
        """Create an LLM provider instance.

        Args:
            model_identifier: Either "provider/model" format (e.g., "openai/gpt-4")
                            or just model name if it's in MODEL_TO_PROVIDER mapping
            config: LLMConfig object or dictionary of config parameters
            **kwargs: Additional configuration parameters

        Returns:
            Configured LLM provider instance

        Raises:
            ValueError: If provider or model is not supported
        """
        provider_name: Optional[str]
        if "/" in model_identifier:
            provider_name, model_name = model_identifier.split("/", 1)
        else:
            provider_name = self.MODEL_TO_PROVIDER.get(model_identifier)
            if not provider_name:
                if "/" in model_identifier or model_identifier in [
                    "gpt2",
                    "distilgpt2",
                ]:
                    provider_name = "huggingface"
                else:
                    raise ValueError(
                        f"Could not determine provider for model '{model_identifier}'. "
                        f"Use 'provider/model' format or check supported models."
                    )
            model_name = model_identifier

        if provider_name not in self.PROVIDERS:
            raise ValueError(
                f"Unsupported provider '{provider_name}'. "
                f"Supported providers: {', '.join(self.PROVIDERS.keys())}"
            )

        if isinstance(config, dict):
            config_dict = config.copy()
        elif isinstance(config, LLMConfig):
            config_dict = config.model_dump()
        else:
            config_dict = {}

        config_dict.update(kwargs)

        provider_class = self.PROVIDERS[provider_name]
        if hasattr(provider_class, "create_config"):
            model_for_config = config_dict.pop("model_name", model_name)
            provider_name_from_dict = config_dict.pop("provider_name", None)
            llm_config = provider_class.create_config(
                model=model_for_config, **config_dict
            )
        else:
            config_dict["provider_name"] = provider_name
            config_dict["model_name"] = model_name
            llm_config = LLMConfig(**config_dict)

        provider_class = self.PROVIDERS[provider_name]
        temp_provider = provider_class.__new__(provider_class)
        temp_provider.config = llm_config
        validation_errors = temp_provider.validate_config()

        if validation_errors:
            raise ValueError(
                f"Configuration validation failed: {'; '.join(validation_errors)}"
            )

        return provider_class(llm_config)

    @classmethod
    def create_from_string(self, model_string: str, **kwargs: Any) -> BaseLLMProvider:
        """Create provider from simple string specification.

        Args:
            model_string: Model specification like "openai/gpt-4", "gpt-4", "claude-3.5-sonnet"
            **kwargs: Additional configuration parameters

        Returns:
            Configured LLM provider instance
        """
        return self.create_provider(model_string, **kwargs)

    @classmethod
    def list_supported_models(self) -> Dict[str, list]:
        """List all supported models by provider.

        Returns:
            Dictionary mapping provider names to lists of supported models
        """
        models_by_provider = {}

        for provider_name, provider_class in self.PROVIDERS.items():
            if hasattr(provider_class, "SUPPORTED_MODELS"):
                models_by_provider[provider_name] = provider_class.SUPPORTED_MODELS
            else:
                models_by_provider[provider_name] = []

        return models_by_provider

    @classmethod
    def get_provider_for_model(self, model_name: str) -> Optional[str]:
        """Get the provider name for a given model.

        Args:
            model_name: Name of the model

        Returns:
            Provider name or None if not found
        """
        return self.MODEL_TO_PROVIDER.get(model_name)

    @classmethod
    def create_config_for_provider(
        self, provider_name: str, model_name: str, **kwargs: Any
    ) -> LLMConfig:
        """Create a default configuration for a specific provider.

        Args:
            provider_name: Name of the provider
            model_name: Name of the model
            **kwargs: Additional configuration parameters

        Returns:
            LLMConfig object with provider-specific defaults
        """
        if provider_name not in self.PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider_name}")

        provider_class = self.PROVIDERS[provider_name]

        if hasattr(provider_class, "create_config"):
            retval: LLMConfig = provider_class.create_config(model=model_name, **kwargs)
            return retval
        else:
            return LLMConfig(
                provider_name=provider_name, model_name=model_name, **kwargs
            )

    @classmethod
    def test_provider_availability(self, provider_name: str) -> tuple[bool, str]:
        """Test if a provider is available (dependencies installed, etc.).

        Args:
            provider_name: Name of the provider to test

        Returns:
            Tuple of (is_available, status_message)
        """
        if provider_name not in self.PROVIDERS:
            return False, f"Unknown provider: {provider_name}"

        provider_class = self.PROVIDERS[provider_name]

        try:
            test_config = LLMConfig(
                provider_name=provider_name, model_name="test-model", api_key="test-key"
            )

            provider_class(test_config)
            return True, f"Provider {provider_name} is available"

        except ImportError as e:
            return False, f"Missing dependencies for {provider_name}: {e}"
        except Exception as e:
            if "not installed" in str(e).lower() or "import" in str(e).lower():
                return False, f"Dependency issue for {provider_name}: {e}"
            else:
                return (
                    True,
                    f"Provider {provider_name} is available (config validation failed as expected)",
                )
