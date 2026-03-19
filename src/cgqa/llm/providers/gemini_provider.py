"""Google Gemini LLM provider implementation."""

import os
from typing import Any, Dict, List, Optional

from .base import BaseLLMProvider, LLMConfig, LLMResponse


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider using the official Google Gen AI SDK."""

    # Verified working Gemini models (1M input, 65K output)
    SUPPORTED_MODELS = [
        # Gemini 3.0 family - Preview models
        "gemini-3-flash-preview",  # Gemini 3 Flash preview
        "gemini-3-pro-preview",  # Gemini 3 Pro preview
        # Gemini 2.5 family - Latest generation (Jan 2025 cutoff)
        "gemini-2.5-pro",  # Complex reasoning, long context
        "gemini-2.5-flash",  # Best price-performance
        "gemini-2.5-flash-lite",  # Cost-efficient, low latency
        # Gemini 2.0 family (Jan 2025 cutoff)
        "gemini-2.0-flash",  # Previous generation Flash
    ]

    def __init__(self, config: LLMConfig):

        try:
            from google import genai  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "Google Gen AI package not installed. Install with: pip install google-genai"
            )

        # Set defaults for Gemini
        if not config.api_key:
            config.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

        if not config.api_key:
            raise ValueError(
                "Google API key not provided. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable "
                "or pass api_key in config."
            )

        super().__init__(config)

    def _setup_client(self) -> None:
        """Initialize the Gemini client."""
        try:
            from google import genai
            from google.genai import types  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "Google Gen AI package not installed. Install with: pip install google-genai"
            )

        # Initialize the client with API key
        self.client = genai.Client(api_key=self.config.api_key)

        # Configure generation parameters using new API
        self.generation_config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
        )

        # Configure safety settings to be less restrictive
        # This prevents false positives on technical/academic content
        self.safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"
            ),
        ]

        # Store model name for use in requests
        self.model_name = self.config.model_name

    def _make_request(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Make a request to Gemini API using new google.genai SDK."""

        try:
            from google.genai import types

            # Check if this is a Gemini 3 reasoning model
            is_gemini_3 = self.model_name.startswith("gemini-3")

            # Set temperature - Gemini 3 uses 1.0 by default, others use config value
            if is_gemini_3 and "temperature" not in kwargs:
                temperature = 1.0
            else:
                temperature = kwargs.get("temperature", self.config.temperature)

            # Create generation config with overrides
            config_kwargs = {
                "temperature": temperature,
                "safety_settings": self.safety_settings,
            }

            # Add thinking configuration for Gemini 3 models
            if is_gemini_3:
                # Default: use "low" for most models (matching OpenAI reasoning model pattern)
                thinking_level = "low"

                # Flash models use "minimal" for faster responses (like gpt-5-mini/nano)
                if "flash" in self.model_name:
                    thinking_level = "minimal"

                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_level=thinking_level
                )

            # Only set max_output_tokens if explicitly provided
            # Check if max_tokens is in kwargs first, otherwise use config
            if "max_tokens" in kwargs:
                max_tokens_value = kwargs["max_tokens"]
            else:
                max_tokens_value = self.config.max_tokens

            # Only add to config if not None (let API use default otherwise)
            if max_tokens_value is not None:
                config_kwargs["max_output_tokens"] = max_tokens_value

            # Add additional generation parameters if provided
            if "top_p" in kwargs:
                config_kwargs["top_p"] = kwargs["top_p"]
            if "top_k" in kwargs:
                config_kwargs["top_k"] = kwargs["top_k"]
            if "stop_sequences" in kwargs:
                config_kwargs["stop_sequences"] = kwargs["stop_sequences"]

            generation_config = types.GenerateContentConfig(**config_kwargs)

            # Generate response using new API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=generation_config,
            )

            # Extract response text with better error handling and debugging
            response_text = ""
            extraction_error = None

            try:
                if hasattr(response, "text") and response.text:
                    response_text = response.text
            except (AttributeError, ValueError, TypeError) as e:
                # If direct text access fails, try candidates
                extraction_error = f"text access failed: {e}"

            # If still no text, try extracting from candidates
            if not response_text:
                try:
                    if hasattr(response, "candidates") and response.candidates:
                        for candidate in response.candidates:
                            if hasattr(candidate, "content") and candidate.content:
                                if (
                                    hasattr(candidate.content, "parts")
                                    and candidate.content.parts
                                ):
                                    for part in candidate.content.parts:
                                        if hasattr(part, "text") and part.text:
                                            response_text += part.text
                except (AttributeError, TypeError) as e:
                    if extraction_error:
                        extraction_error += f" | candidates access failed: {e}"
                    else:
                        extraction_error = f"candidates access failed: {e}"

            # If we still have no text but no explicit error, check response structure
            if not response_text and not extraction_error:
                # Try to understand why we got no text
                response_str = str(response)[:200] if response else "None"
                extraction_error = f"No text extracted from response: {response_str}"

            # Handle usage metadata
            usage_metadata = getattr(response, "usage_metadata", None)
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None

            if usage_metadata:
                prompt_tokens = getattr(usage_metadata, "prompt_token_count", None)
                completion_tokens = getattr(
                    usage_metadata, "candidates_token_count", None
                )
                total_tokens = getattr(usage_metadata, "total_token_count", None)

            # If no text was extracted, include the error
            if not response_text and extraction_error:
                return LLMResponse(
                    text="",
                    model_name=self.config.model_name,
                    provider_name="gemini",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    raw_response=self._response_to_dict(response),
                    error=extraction_error,
                    error_type="ResponseExtractionError",
                )

            return LLMResponse(
                text=response_text,
                model_name=self.config.model_name,
                provider_name="gemini",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                raw_response=self._response_to_dict(response),
            )

        except Exception as e:
            return LLMResponse(
                text="",
                model_name=self.config.model_name,
                provider_name="gemini",
                error=str(e),
                error_type=type(e).__name__,
            )

    def _response_to_dict(self, response: Any) -> Dict[str, Any]:
        """Convert Gemini response to dictionary for raw_response field."""
        try:
            result = {
                "text": getattr(response, "text", ""),
                "finish_reason": (
                    getattr(response.candidates[0], "finish_reason", None)
                    if response.candidates
                    else None
                ),
            }

            if hasattr(response, "usage_metadata"):
                result["usage_metadata"] = {
                    "prompt_token_count": getattr(
                        response.usage_metadata, "prompt_token_count", None
                    ),
                    "candidates_token_count": getattr(
                        response.usage_metadata, "candidates_token_count", None
                    ),
                    "total_token_count": getattr(
                        response.usage_metadata, "total_token_count", None
                    ),
                }

            return result
        except Exception:
            return {"error": "Failed to parse response"}

    def test_connection(self) -> bool:
        """Test connection to Gemini API."""
        try:
            # Make a minimal request using new API
            # Don't specify max_tokens - let model use its default
            response = self.generate("Hello", temperature=0.1)
            return response.error is None
        except Exception:
            return False

    def validate_config(self) -> List[str]:
        """Validate Gemini-specific configuration."""
        errors = super().validate_config()

        if not self.config.api_key:
            errors.append("api_key is required for Gemini provider")

        if self.config.model_name not in self.SUPPORTED_MODELS:
            errors.append(
                f"Unsupported model '{self.config.model_name}'. "
                f"Supported models: {', '.join(self.SUPPORTED_MODELS)}"
            )

        return errors

    def get_model_info(self) -> Dict[str, Any]:
        """Get Gemini model information."""
        info = super().get_model_info()
        info.update(
            {
                "provider_specific": {
                    "api_version": "v1",
                    "supports_system_messages": True,
                    "supports_multimodal": self._supports_multimodal(),
                    "context_window": self._get_context_window(),
                    "training_data_cutoff": self._get_training_cutoff(),
                    "safety_settings": True,
                }
            }
        )
        return info

    def _supports_multimodal(self) -> bool:
        """Check if model supports multimodal input."""
        multimodal_models = [
            "gemini-1.5-pro",
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.0-pro-vision",
        ]
        return self.config.model_name in multimodal_models

    def _get_context_window(self) -> int:
        """Get context window size for the model."""
        context_windows = {
            # Gemini 3.0 family - 1M input tokens (preview models)
            "gemini-3-flash-preview": 1048576,
            "gemini-3-pro-preview": 1048576,
            # Gemini 2.5 family - 1M input tokens
            "gemini-2.5-pro": 1048576,
            "gemini-2.5-flash": 1048576,
            "gemini-2.5-flash-lite": 1048576,
            # Gemini 2.0 family - 1M input tokens
            "gemini-2.0-flash": 1048576,
            "gemini-2.0-flash-lite": 1048576,
            # Gemini 1.5 family - 1M tokens
            "gemini-1.5-pro": 1048576,
            "gemini-1.5-pro-latest": 1048576,
            "gemini-1.5-flash": 1048576,
            "gemini-1.5-flash-latest": 1048576,
            # Gemini 1.0 family
            "gemini-1.0-pro": 30720,
            "gemini-1.0-pro-latest": 30720,
            # Experimental models - 2M tokens
            "gemini-exp-1114": 2097152,
            "gemini-exp-1121": 2097152,
        }
        return context_windows.get(self.config.model_name, 30720)

    def _get_training_cutoff(self) -> str:
        """Get training data cutoff for the model."""
        cutoffs = {
            # Gemini 3.0 family - Preview models
            "gemini-3-flash-preview": "January 2025",
            "gemini-3-pro-preview": "January 2025",
            # Gemini 2.5 family - Knowledge cutoff: January 2025
            "gemini-2.5-pro": "January 2025",
            "gemini-2.5-flash": "January 2025",
            "gemini-2.5-flash-lite": "January 2025",
            # Gemini 2.0 family - Knowledge cutoff: January 2025
            "gemini-2.0-flash": "January 2025",
            "gemini-2.0-flash-lite": "January 2025",
            # Gemini 1.5 family
            "gemini-1.5-pro": "April 2024",
            "gemini-1.5-pro-latest": "April 2024",
            "gemini-1.5-flash": "April 2024",
            "gemini-1.5-flash-latest": "April 2024",
            # Gemini 1.0 family
            "gemini-1.0-pro": "February 2024",
            "gemini-1.0-pro-latest": "February 2024",
            # Experimental models
            "gemini-exp-1114": "November 2024",
            "gemini-exp-1121": "November 2024",
        }
        return cutoffs.get(self.config.model_name, "Unknown")

    def format_question_prompt(
        self,
        question: str,
        context: Optional[str] = None,
        answer_type: Optional[str] = None,
        question_type: Optional[str] = None,
    ) -> str:
        """Format a question into a prompt suitable for Gemini.

        Gemini works well with clear instructions and structured input.
        """
        # Use the base class implementation for structured prompting
        return super().format_question_prompt(
            question, context, answer_type, question_type  # type: ignore
        )

    @classmethod
    def create_config(
        cls,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMConfig:
        """Create a configuration for Gemini provider.

        Args:
            model: Gemini model name
            api_key: API key (uses GOOGLE_API_KEY or GEMINI_API_KEY env var if not provided)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional configuration parameters

        Returns:
            LLMConfig object
        """
        return LLMConfig(
            provider_name="gemini",
            model_name=model,
            api_key=api_key
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
