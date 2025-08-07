"""HuggingFace LLM provider implementation."""

import os
import time
import json
from typing import Dict, Any, Optional, List
from .base import BaseLLMProvider, LLMResponse, LLMConfig

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace LLM provider supporting both API and local inference."""
    
    # Popular open-source models available on HuggingFace
    SUPPORTED_MODELS = [
        # Llama models
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf", 
        "meta-llama/Llama-2-70b-chat-hf",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Meta-Llama-3-70B-Instruct",
        
        # Mistral models
        "mistralai/Mistral-7B-Instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        
        # Other popular models
        "microsoft/DialoGPT-medium",
        "microsoft/DialoGPT-large",
        "EleutherAI/gpt-j-6b",
        "EleutherAI/gpt-neox-20b",
        "bigscience/bloom-7b1",
        "google/flan-t5-large",
        "google/flan-t5-xl",
        
        # Coding models
        "codellama/CodeLlama-7b-Instruct-hf",
        "codellama/CodeLlama-13b-Instruct-hf",
        
        # Smaller models for testing
        "distilgpt2",
        "gpt2",
        "gpt2-medium",
    ]
    
    def __init__(self, config: LLMConfig):
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "Requests package not installed. Install with: pip install requests"
            )
        
        # Determine inference mode: 'api' or 'local'
        self.inference_mode = config.extra_params.get("inference_mode", "api")
        
        if self.inference_mode == "local" and not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers package not installed for local inference. "
                "Install with: pip install transformers torch"
            )
        
        # Set defaults for HuggingFace
        if self.inference_mode == "api":
            if not config.api_key:
                config.api_key = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
            
            if not config.api_base:
                config.api_base = "https://api-inference.huggingface.co/models"
        
        super().__init__(config)
    
    def _setup_client(self) -> None:
        """Initialize the HuggingFace client."""
        if self.inference_mode == "api":
            self._setup_api_client()
        else:
            self._setup_local_client()
    
    def _setup_api_client(self) -> None:
        """Setup API-based inference."""
        self.api_url = f"{self.config.api_base}/{self.config.model_name}"
        self.headers = {}
        
        if self.config.api_key:
            self.headers["Authorization"] = f"Bearer {self.config.api_key}"
    
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
                self.config.model_name,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if device == "cpu":
                self.model = self.model.to(device)
            
            self.device = device
            
        except Exception as e:
            raise ValueError(f"Failed to load HuggingFace model '{self.config.model_name}': {e}")
    
    def _make_request(self, prompt: str, **kwargs) -> LLMResponse:
        """Make a request to HuggingFace (API or local)."""
        if self.inference_mode == "api":
            return self._make_api_request(prompt, **kwargs)
        else:
            return self._make_local_request(prompt, **kwargs)
    
    def _make_api_request(self, prompt: str, **kwargs) -> LLMResponse:
        """Make an API request to HuggingFace Inference API."""
        try:
            # Prepare payload
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "return_full_text": False,
                }
            }
            
            # Add additional parameters
            for key, value in kwargs.items():
                if key not in ["max_tokens", "temperature"] and not key.startswith("_"):
                    if key in ["top_p", "top_k", "repetition_penalty", "do_sample"]:
                        payload["parameters"][key] = value
            
            # Make request
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=self.config.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Handle different response formats
            response_text = ""
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    response_text = result[0]["generated_text"]
                elif "text" in result[0]:
                    response_text = result[0]["text"]
            elif isinstance(result, dict):
                if "generated_text" in result:
                    response_text = result["generated_text"]
                elif "text" in result:
                    response_text = result["text"]
            
            return LLMResponse(
                text=response_text,
                model_name=self.config.model_name,
                provider_name="huggingface",
                raw_response=result
            )
        
        except Exception as e:
            return LLMResponse(
                text="",
                model_name=self.config.model_name,
                provider_name="huggingface",
                error=str(e),
                error_type=type(e).__name__
            )
    
    def _make_local_request(self, prompt: str, **kwargs) -> LLMResponse:
        """Make a local inference request."""
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generation parameters
            generation_kwargs = {
                "max_new_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "do_sample": True if kwargs.get("temperature", self.config.temperature) > 0 else False,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            
            # Add additional parameters
            for key, value in kwargs.items():
                if key not in ["max_tokens", "temperature"] and not key.startswith("_"):
                    if key in ["top_p", "top_k", "repetition_penalty", "num_beams"]:
                        generation_kwargs[key] = value
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            # Decode response
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
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
                total_tokens=total_tokens
            )
        
        except Exception as e:
            return LLMResponse(
                text="",
                model_name=self.config.model_name,
                provider_name="huggingface",
                error=str(e),
                error_type=type(e).__name__
            )
    
    def test_connection(self) -> bool:
        """Test connection to HuggingFace."""
        try:
            response = self.generate("Hello", max_tokens=1, temperature=0.1)
            return response.error is None
        except Exception:
            return False
    
    def validate_config(self) -> List[str]:
        """Validate HuggingFace-specific configuration."""
        errors = super().validate_config()
        
        if self.inference_mode == "api" and not self.config.api_key:
            errors.append(
                "api_key or HF_TOKEN environment variable is recommended for HuggingFace API access"
            )
        
        if not self.config.model_name:
            errors.append("model_name is required")
        
        # Note: We don't strictly validate model names since HF has thousands of models
        
        return errors
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get HuggingFace model information."""
        info = super().get_model_info()
        info.update({
            "provider_specific": {
                "inference_mode": self.inference_mode,
                "supports_local_inference": TRANSFORMERS_AVAILABLE,
                "supports_api_inference": True,
                "hub_url": f"https://huggingface.co/{self.config.model_name}",
                "device": getattr(self, 'device', 'unknown') if self.inference_mode == "local" else "api",
            }
        })
        return info
    
    def format_question_prompt(self, question: str, context: Optional[str] = None, answer_type: Optional[str] = None) -> str:
        """Format a question into a prompt suitable for HuggingFace models.
        
        Many HF models work better with chat-style prompts, but we also need structured output.
        """
        # Get the structured prompt from base class
        base_prompt = super().format_question_prompt(question, context, answer_type)
        
        # Check if model is a chat/instruct model
        is_chat_model = any(keyword in self.config.model_name.lower() 
                           for keyword in ["chat", "instruct", "dialogue"])
        
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
        max_tokens: int = 1000,
        **kwargs
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
            api_key=api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
            extra_params=extra_params
        )