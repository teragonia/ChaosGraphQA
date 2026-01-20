"""HuggingFace LLM provider implementation."""

import json
import os
import time
from typing import Any, Dict, List, Optional

from .base import BaseLLMProvider, LLMConfig, LLMResponse

try:
    from huggingface_hub import InferenceClient

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace LLM provider supporting both API and local inference."""

    # Models verified to work with HuggingFace InferenceClient (hf-inference provider)
    # Note: This list contains models tested and working. Other models from HuggingFace
    # may work if they support the chat completions interface on hf-inference.
    # For local inference, set inference_mode: "local" in config.
    SUPPORTED_MODELS = [
        # SmolLM models (128k context, multilingual, reasoning)
        "HuggingFaceTB/SmolLM3-3B",
        # Note: Most models require local inference mode or may not be available
        # via the hf-inference provider. To use other models:
        # 1. Try adding them here and test with: cgqa test-model --model huggingface/model-name
        # 2. Or use inference_mode: "local" in your config for local transformers inference
    ]

    def __init__(self, config: LLMConfig):
        # Ensure extra_params exists
        extra_params: dict = {} if config.extra_params is None else config.extra_params

        # Determine inference mode: 'api' or 'local'
        # Set this BEFORE calling super().__init__() so it's available in all methods
        self.inference_mode = extra_params.get("inference_mode", "api")

        if self.inference_mode == "api" and not HF_HUB_AVAILABLE:
            raise ImportError(
                "huggingface_hub package not installed. Install with: pip install huggingface_hub"
            )

        if self.inference_mode == "local" and not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers package not installed for local inference. "
                "Install with: pip install transformers torch"
            )

        # Set defaults for HuggingFace
        if self.inference_mode == "api":
            if not config.api_key:
                config.api_key = os.getenv("HF_TOKEN") or os.getenv(
                    "HUGGINGFACE_API_KEY"
                )

        # Call parent init after setting inference_mode
        super().__init__(config)

    def _setup_client(self) -> None:
        """Initialize the HuggingFace client."""
        if self.inference_mode == "api":
            self._setup_api_client()
        else:
            self._setup_local_client()

    def _setup_api_client(self) -> None:
        """Setup API-based inference using InferenceClient."""
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=self.config.api_key,
        )

    def _setup_local_client(self) -> None:
        """Setup local inference with transformers."""
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load model with appropriate settings for device
            model_kwargs = {
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
                "device_map": "auto" if device == "cuda" else None,
            }

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name, **model_kwargs
            )

            # Move to device if not using device_map
            if device == "cpu":
                self.model = self.model.to(device)

            self.device = device

        except Exception as e:
            raise ValueError(
                f"Failed to load HuggingFace model '{self.config.model_name}': {e}"
            )

    def _make_request(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Make a request to HuggingFace (API or local)."""
        if self.inference_mode == "api":
            return self._make_api_request(prompt, **kwargs)
        else:
            return self._make_local_request(prompt, **kwargs)

    def _make_api_request(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Make an API request to HuggingFace Inference API using InferenceClient."""
        try:
            # Prepare request parameters for chat completions
            request_params = {
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", self.config.temperature),
            }

            # Only include max_tokens if explicitly set (not None)
            max_tokens_value = kwargs.get("max_tokens", self.config.max_tokens)
            if max_tokens_value is not None:
                request_params["max_tokens"] = max_tokens_value

            # Add additional parameters
            for key, value in kwargs.items():
                if key not in ["max_tokens", "temperature"] and not key.startswith("_"):
                    if key in [
                        "top_p",
                        "top_k",
                        "frequency_penalty",
                        "presence_penalty",
                    ]:
                        request_params[key] = value

            # Make request using chat completions
            response = self.client.chat.completions.create(**request_params)

            # Extract response text
            response_text = ""
            if response.choices and len(response.choices) > 0:
                response_text = response.choices[0].message.content or ""

            # Handle usage data if available
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
            completion_tokens = (
                getattr(usage, "completion_tokens", None) if usage else None
            )
            total_tokens = getattr(usage, "total_tokens", None) if usage else None

            return LLMResponse(
                text=response_text,
                model_name=self.config.model_name,
                provider_name="huggingface",
                request_id=getattr(response, "id", None),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        except Exception as e:
            return LLMResponse(
                text="",
                model_name=self.config.model_name,
                provider_name="huggingface",
                error=str(e),
                error_type=type(e).__name__,
            )

    def _make_local_request(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Make a local inference request."""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generation parameters
            generation_kwargs = {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "do_sample": (
                    True
                    if kwargs.get("temperature", self.config.temperature) > 0
                    else False
                ),
                "pad_token_id": self.tokenizer.eos_token_id,
            }

            # Only include max_new_tokens if explicitly set (not None)
            max_tokens_value = kwargs.get("max_tokens", self.config.max_tokens)
            if max_tokens_value is not None:
                generation_kwargs["max_new_tokens"] = max_tokens_value

            # Add additional parameters
            for key, value in kwargs.items():
                if key not in ["max_tokens", "temperature"] and not key.startswith("_"):
                    if key in ["top_p", "top_k", "repetition_penalty", "num_beams"]:
                        generation_kwargs[key] = value

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)

            # Decode response
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response_text = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )

            # Calculate token counts
            prompt_tokens = inputs["input_ids"].shape[1]
            completion_tokens = len(generated_tokens)
            total_tokens = prompt_tokens + completion_tokens

            return LLMResponse(
                text=response_text,
                model_name=self.config.model_name,
                provider_name="huggingface",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        except Exception as e:
            return LLMResponse(
                text="",
                model_name=self.config.model_name,
                provider_name="huggingface",
                error=str(e),
                error_type=type(e).__name__,
            )

    def test_connection(self) -> bool:
        """Test connection to HuggingFace."""
        try:
            # Make a minimal request - don't specify max_tokens, let model use its default
            response = self.generate("Hello", temperature=0.1)
            return response.error is None
        except Exception:
            return False

    def validate_config(self) -> List[str]:
        """Validate HuggingFace-specific configuration."""
        errors = super().validate_config()

        # Get inference mode (might not be set if called from factory validation)
        inference_mode = getattr(self, "inference_mode", None)
        if inference_mode is None and self.config.extra_params:
            inference_mode = self.config.extra_params.get("inference_mode", "api")
        elif inference_mode is None:
            inference_mode = "api"

        if inference_mode == "api" and not self.config.api_key:
            errors.append(
                "api_key or HF_TOKEN environment variable is recommended for HuggingFace API access"
            )

        if not self.config.model_name:
            errors.append("model_name is required")

        # Note: We don't strictly validate model names since HF has thousands of models.
        # Users can try any model - if it doesn't work with hf-inference, they can use local mode.

        return errors

    def get_model_info(self) -> Dict[str, Any]:
        """Get HuggingFace model information."""
        info = super().get_model_info()

        # Add context window info
        context_window = self._get_context_window()
        if context_window:
            info["context_window"] = context_window

        info.update(
            {
                "provider_specific": {
                    "inference_mode": self.inference_mode,
                    "supports_local_inference": TRANSFORMERS_AVAILABLE,
                    "supports_api_inference": True,
                    "hub_url": f"https://huggingface.co/{self.config.model_name}",
                    "context_window": context_window,
                    "device": (
                        getattr(self, "device", "unknown")
                        if self.inference_mode == "local"
                        else "api"
                    ),
                }
            }
        )
        return info

    def _get_context_window(self) -> Optional[int]:
        """Get context window size for the model."""
        context_windows = {
            "HuggingFaceTB/SmolLM3-3B": 128000,
            "openai/gpt-oss-20b": 8192,  # Estimated
        }
        return context_windows.get(self.config.model_name)

    def format_question_prompt(
        self,
        question: str,
        context: Optional[str] = None,
        answer_type: Optional[str] = None,
        question_type: Optional[str] = None,
    ) -> str:
        """Format a question into a prompt suitable for HuggingFace models.

        Many HF models work better with chat-style prompts, but we also need structured output.
        """
        # Get the structured prompt from base class
        base_prompt = super().format_question_prompt(
            question, context, answer_type, question_type  # type: ignore
        )

        # Check if model is a chat/instruct model
        is_chat_model = any(
            keyword in self.config.model_name.lower()
            for keyword in ["chat", "instruct", "dialogue"]
        )

        if is_chat_model:
            # Wrap in chat format
            return f"""<|im_start|>user
{base_prompt}<|im_end|>
<|im_start|>assistant"""
        else:
            # For base models, just add Answer: prefix
            return f"{base_prompt}\n\nAnswer:"

    @classmethod
    def create_config(
        cls,
        model: str = "distilgpt2",
        api_key: Optional[str] = None,
        inference_mode: str = "api",
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMConfig:
        """Create a configuration for HuggingFace provider.

        Args:
            model: HuggingFace model name
            api_key: HF token (uses HF_TOKEN env var if not provided)
            inference_mode: 'api' or 'local'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional configuration parameters

        Returns:
            LLMConfig object
        """
        extra_params = kwargs.copy()
        extra_params["inference_mode"] = inference_mode

        return LLMConfig(
            provider_name="huggingface",
            model_name=model,
            api_key=api_key
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
            extra_params=extra_params,
        )
