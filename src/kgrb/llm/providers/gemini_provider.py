"""Google Gemini LLM provider implementation."""

import os
from typing import Dict, Any, Optional, List
from .base import BaseLLMProvider, LLMResponse, LLMConfig

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider using the official Google AI API."""
    
    # Common Gemini models
    SUPPORTED_MODELS = [
        "gemini-1.5-pro",
        "gemini-1.5-pro-latest", 
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.0-pro",
        "gemini-1.0-pro-latest",
        # Experimental models
        "gemini-exp-1114",
        "gemini-exp-1121"
    ]
    
    def __init__(self, config: LLMConfig):
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Google AI package not installed. Install with: pip install google-generativeai"
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
        genai.configure(api_key=self.config.api_key)
        
        # Configure generation parameters
        self.generation_config = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
        }
        
        # Initialize the model
        try:
            self.model = genai.GenerativeModel(
                model_name=self.config.model_name,
                generation_config=self.generation_config
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini model '{self.config.model_name}': {e}")
    
    def _make_request(self, prompt: str, **kwargs) -> LLMResponse:
        """Make a request to Gemini API."""
        
        try:
            # Update generation config with any overrides
            generation_config = self.generation_config.copy()
            if "max_tokens" in kwargs:
                generation_config["max_output_tokens"] = kwargs["max_tokens"]
            if "temperature" in kwargs:
                generation_config["temperature"] = kwargs["temperature"]
            
            # Add additional generation parameters
            for key, value in kwargs.items():
                if key not in ["max_tokens", "temperature"] and not key.startswith("_"):
                    if key in ["top_p", "top_k", "candidate_count", "stop_sequences"]:
                        generation_config[key] = value
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Extract response text
            response_text = ""
            if response.text:
                response_text = response.text
            elif response.parts:
                # Sometimes response comes as parts
                for part in response.parts:
                    if hasattr(part, 'text'):
                        response_text += part.text
            
            # Handle usage data (if available)
            usage_metadata = getattr(response, 'usage_metadata', None)
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None
            
            if usage_metadata:
                prompt_tokens = getattr(usage_metadata, 'prompt_token_count', None)
                completion_tokens = getattr(usage_metadata, 'candidates_token_count', None)
                total_tokens = getattr(usage_metadata, 'total_token_count', None)
            
            return LLMResponse(
                text=response_text,
                model_name=self.config.model_name,
                provider_name="gemini",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                raw_response=self._response_to_dict(response)
            )
        
        except Exception as e:
            return LLMResponse(
                text="",
                model_name=self.config.model_name,
                provider_name="gemini",
                error=str(e),
                error_type=type(e).__name__
            )
    
    def _response_to_dict(self, response) -> Dict[str, Any]:
        """Convert Gemini response to dictionary for raw_response field."""
        try:
            result = {
                "text": getattr(response, 'text', ''),
                "finish_reason": getattr(response.candidates[0], 'finish_reason', None) if response.candidates else None,
            }
            
            if hasattr(response, 'usage_metadata'):
                result["usage_metadata"] = {
                    "prompt_token_count": getattr(response.usage_metadata, 'prompt_token_count', None),
                    "candidates_token_count": getattr(response.usage_metadata, 'candidates_token_count', None),
                    "total_token_count": getattr(response.usage_metadata, 'total_token_count', None),
                }
            
            return result
        except Exception:
            return {"error": "Failed to parse response"}
    
    def test_connection(self) -> bool:
        """Test connection to Gemini API."""
        try:
            # Make a minimal request
            response = self.generate("Hello", max_tokens=1, temperature=0.1)
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
        info.update({
            "provider_specific": {
                "api_version": "v1",
                "supports_system_messages": True,
                "supports_multimodal": self._supports_multimodal(),
                "context_window": self._get_context_window(),
                "training_data_cutoff": self._get_training_cutoff(),
                "safety_settings": True,
            }
        })
        return info
    
    def _supports_multimodal(self) -> bool:
        """Check if model supports multimodal input."""
        multimodal_models = [
            "gemini-1.5-pro", "gemini-1.5-pro-latest",
            "gemini-1.5-flash", "gemini-1.5-flash-latest", 
            "gemini-1.0-pro-vision"
        ]
        return self.config.model_name in multimodal_models
    
    def _get_context_window(self) -> int:
        """Get context window size for the model."""
        context_windows = {
            "gemini-1.5-pro": 1048576,  # 1M tokens
            "gemini-1.5-pro-latest": 1048576,
            "gemini-1.5-flash": 1048576,
            "gemini-1.5-flash-latest": 1048576,
            "gemini-1.0-pro": 30720,
            "gemini-1.0-pro-latest": 30720,
            "gemini-exp-1114": 2097152,  # 2M tokens
            "gemini-exp-1121": 2097152,
        }
        return context_windows.get(self.config.model_name, 30720)
    
    def _get_training_cutoff(self) -> str:
        """Get training data cutoff for the model."""
        cutoffs = {
            "gemini-1.5-pro": "April 2024",
            "gemini-1.5-pro-latest": "April 2024",
            "gemini-1.5-flash": "April 2024",
            "gemini-1.5-flash-latest": "April 2024",
            "gemini-1.0-pro": "February 2024",
            "gemini-1.0-pro-latest": "February 2024",
            "gemini-exp-1114": "November 2024",
            "gemini-exp-1121": "November 2024",
        }
        return cutoffs.get(self.config.model_name, "February 2024")
    
    def format_question_prompt(self, question: str, context: Optional[str] = None) -> str:
        """Format a question into a prompt suitable for Gemini.
        
        Gemini works well with clear instructions and structured input.
        """
        if context:
            return f"""Please analyze the following context and answer the question based on the information provided.

Context:
{context}

Question: {question}

Please provide a clear, direct answer based on the context. If you need to show your reasoning, please do so step by step."""
        else:
            return f"""Question: {question}

Please provide a clear, direct answer."""
    
    @classmethod
    def create_config(
        cls,
        model: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        **kwargs
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
            api_key=api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )