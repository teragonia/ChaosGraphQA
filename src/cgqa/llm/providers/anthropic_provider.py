"""Anthropic LLM provider implementation."""

import os
from typing import Dict, Any, Optional, List
from .base import BaseLLMProvider, LLMResponse, LLMConfig

try:
    import anthropic
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None


class AnthropicProvider(BaseLLMProvider):
    """Anthropic LLM provider using the official Anthropic API."""
    
    # Common Anthropic models
    SUPPORTED_MODELS = [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620", 
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        # Aliases for convenience
        "claude-4-sonnet",
        "claude-3.5-sonnet",
        "claude-3.5-haiku",
        "claude-3-opus",
        "claude-3-sonnet", 
        "claude-3-haiku",
    ]
    
    # Model name mappings for aliases
    MODEL_ALIASES = {
        "claude-4-sonnet": "claude-sonnet-4-20250514",
        "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3.5-haiku": "claude-3-5-haiku-20241022",
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-haiku": "claude-3-haiku-20240307"
    }
    
    def __init__(self, config: LLMConfig):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic package not installed. Install with: pip install anthropic"
            )
        
        # Set defaults for Anthropic
        if not config.api_key:
            config.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not config.api_key:
            raise ValueError(
                "Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key in config."
            )
        
        # Resolve model aliases
        if config.model_name in self.MODEL_ALIASES:
            config.model_name = self.MODEL_ALIASES[config.model_name]
        
        super().__init__(config)
    
    def _setup_client(self) -> None:
        """Initialize the Anthropic client."""
        client_kwargs = {
            "api_key": self.config.api_key,
            "timeout": self.config.timeout,
        }
        
        if self.config.api_base:
            client_kwargs["base_url"] = self.config.api_base
        
        self.client = Anthropic(**client_kwargs)
    
    def _make_request(self, prompt: str, **kwargs) -> LLMResponse:
        """Make a request to Anthropic API."""
        
        # Prepare request parameters
        request_params = {
            "model": self.config.model_name,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "messages": [{"role": "user", "content": prompt}]
        }
        
        # Add any extra parameters
        for key, value in kwargs.items():
            if key not in ["max_tokens", "temperature"] and not key.startswith("_"):
                if key in ["top_p", "top_k", "stop_sequences", "system"]:
                    request_params[key] = value
        
        try:
            response = self.client.messages.create(**request_params)
            
            # Extract response data
            response_text = ""
            if response.content and len(response.content) > 0:
                # Claude returns content as a list of content blocks
                for content_block in response.content:
                    if hasattr(content_block, 'text'):
                        response_text += content_block.text
            
            # Handle usage data
            usage = response.usage if hasattr(response, 'usage') and response.usage else None
            
            return LLMResponse(
                text=response_text,
                model_name=self.config.model_name,
                provider_name="anthropic",
                request_id=response.id if hasattr(response, 'id') else None,
                prompt_tokens=usage.input_tokens if usage else None,
                completion_tokens=usage.output_tokens if usage else None,
                total_tokens=(usage.input_tokens + usage.output_tokens) if usage else None,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
            )
        
        except Exception as e:
            return LLMResponse(
                text="",
                model_name=self.config.model_name,
                provider_name="anthropic",
                error=str(e),
                error_type=type(e).__name__
            )
    
    def test_connection(self) -> bool:
        """Test connection to Anthropic API."""
        try:
            # Make a minimal request
            response = self.generate("Hello", max_tokens=1, temperature=0.1)
            return response.error is None
        except Exception:
            return False
    
    def validate_config(self) -> List[str]:
        """Validate Anthropic-specific configuration."""
        errors = super().validate_config()
        
        if not self.config.api_key:
            errors.append("api_key is required for Anthropic provider")
        
        # Check model name (including aliases)
        model_name = self.config.model_name
        if model_name in self.MODEL_ALIASES:
            model_name = self.MODEL_ALIASES[model_name]
        
        if model_name not in self.SUPPORTED_MODELS:
            errors.append(
                f"Unsupported model '{self.config.model_name}'. "
                f"Supported models: {', '.join(self.SUPPORTED_MODELS[:8])}..."  # Show first few
            )
        
        return errors
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Anthropic model information."""
        info = super().get_model_info()
        info.update({
            "provider_specific": {
                "api_version": "2023-06-01",
                "supports_system_messages": True,
                "supports_function_calling": False,  # Claude doesn't have function calling like OpenAI
                "context_window": self._get_context_window(),
                "training_data_cutoff": self._get_training_cutoff(),
            }
        })
        return info
    
    def _get_context_window(self) -> int:
        """Get context window size for the model."""
        context_windows = {
            "claude-3-5-sonnet-20241022": 200000,
            "claude-3-5-sonnet-20240620": 200000,
            "claude-3-5-haiku-20241022": 200000,
            "claude-3-opus-20240229": 200000,
            "claude-3-sonnet-20240229": 200000,
            "claude-3-haiku-20240307": 200000,
        }
        return context_windows.get(self.config.model_name, 200000)
    
    def _get_training_cutoff(self) -> str:
        """Get training data cutoff for the model."""
        cutoffs = {
            "claude-3-5-sonnet-20241022": "April 2024",
            "claude-3-5-sonnet-20240620": "April 2024",
            "claude-3-5-haiku-20241022": "July 2024",
            "claude-3-opus-20240229": "August 2023",
            "claude-3-sonnet-20240229": "August 2023",
            "claude-3-haiku-20240307": "August 2023",
        }
        return cutoffs.get(self.config.model_name, "August 2023")
    
    def format_question_prompt(self, question: str, context: Optional[str] = None, answer_type: Optional[str] = None) -> str:
        """Format a question into a prompt suitable for Claude.
        
        Claude works well with clear, structured prompts with format instructions.
        """
        # Use the base class implementation for structured prompting
        return super().format_question_prompt(question, context, answer_type)
    
    @classmethod
    def create_config(
        cls,
        model: str = "claude-3.5-sonnet",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMConfig:
        """Create a configuration for Anthropic provider.
        
        Args:
            model: Anthropic model name
            api_key: API key (uses ANTHROPIC_API_KEY env var if not provided)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional configuration parameters
            
        Returns:
            LLMConfig object
        """
        return LLMConfig(
            provider_name="anthropic",
            model_name=model,
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )