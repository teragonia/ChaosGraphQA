"""Base LLM provider interface."""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for LLM providers."""
    
    model_config = {
        "protected_namespaces": (),
        "extra": "allow"
    }
    
    provider_name: str = Field(..., description="Name of the LLM provider")
    model_name: str = Field(..., description="Specific model identifier")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    api_base: Optional[str] = Field(default=None, description="Custom API base URL")
    max_tokens: int = Field(default=1000, description="Maximum tokens to generate")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Sampling temperature")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    rate_limit_delay: float = Field(default=0.1, description="Delay between requests (seconds)")
    
    # Provider-specific settings
    extra_params: Dict[str, Any] = Field(default_factory=dict, description="Additional provider-specific parameters")


@dataclass
class LLMResponse:
    """Standardized response from LLM providers."""
    
    text: str
    model_name: str
    provider_name: str
    timestamp: float = field(default_factory=time.time)
    
    # Token usage (if available)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    
    # Request metadata
    request_id: Optional[str] = None
    response_time: Optional[float] = None  # in seconds
    
    # Raw response for debugging
    raw_response: Optional[Dict[str, Any]] = None
    
    # Error information
    error: Optional[str] = None
    error_type: Optional[str] = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider_name = config.provider_name
        self._setup_client()
    
    @abstractmethod
    def _setup_client(self) -> None:
        """Initialize the provider-specific client."""
        pass
    
    @abstractmethod
    def _make_request(self, prompt: str, **kwargs) -> LLMResponse:
        """Make a request to the LLM provider.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for this request
            
        Returns:
            LLMResponse object with the result
        """
        pass
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response from the LLM.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (merged with config)
            
        Returns:
            LLMResponse object
        """
        start_time = time.time()
        
        try:
            # Apply rate limiting
            if self.config.rate_limit_delay > 0:
                time.sleep(self.config.rate_limit_delay)
            
            # Merge kwargs with config defaults
            request_params = {
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                **self.config.extra_params,
                **kwargs  # Override defaults with explicit parameters
            }
            
            response = self._make_request(prompt, **request_params)
            response.response_time = time.time() - start_time
            
            return response
            
        except Exception as e:
            return LLMResponse(
                text="",
                model_name=self.config.model_name,
                provider_name=self.provider_name,
                error=str(e),
                error_type=type(e).__name__,
                response_time=time.time() - start_time
            )
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[LLMResponse]:
        """Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional parameters
            
        Returns:
            List of LLMResponse objects
        """
        responses = []
        for prompt in prompts:
            response = self.generate(prompt, **kwargs)
            responses.append(response)
        
        return responses
    
    def format_question_prompt(self, question: str, context: Optional[str] = None, answer_type: Optional[str] = None) -> str:
        """Format a question into a prompt suitable for the LLM.
        
        Args:
            question: The question to ask
            context: Optional context (e.g., knowledge graph description)
            answer_type: Expected answer type for format guidance
            
        Returns:
            Formatted prompt string
        """
        # Get format instructions based on answer type
        format_instructions = self._get_format_instructions(answer_type)
        
        base_prompt = f"""Context: {context}

Question: {question}

{format_instructions}"""
        
        return base_prompt
    
    def _get_format_instructions(self, answer_type: Optional[str]) -> str:
        """Get formatting instructions based on answer type."""
        
        if answer_type == "single_entity":
            return """Please provide your answer in exactly this format:
ANSWER: [Entity Name]

Examples:
ANSWER: Project 10
ANSWER: Maintenance 4
ANSWER: Launch 8

Provide only the entity name after "ANSWER:" with no additional text or explanation."""
        
        elif answer_type == "entity_list":
            return """Please provide your answer in exactly this format:
ANSWER: [Entity1, Entity2, Entity3]

Examples:
ANSWER: [Project 10, Launch 8]
ANSWER: [Planning 12, Review 6, Maintenance 4]
ANSWER: []

If there are no entities, use empty brackets: []
Separate multiple entities with commas inside the brackets."""
        
        elif answer_type == "boolean":
            return """Please provide your answer in exactly this format:
ANSWER: [Yes/No]

Examples:
ANSWER: Yes
ANSWER: No

Provide only "Yes" or "No" after "ANSWER:" with no additional text."""
        
        elif answer_type == "numeric":
            return """Please provide your answer in exactly this format:
ANSWER: [Number]

Examples:
ANSWER: 3
ANSWER: 7.5
ANSWER: 0

Provide only the number after "ANSWER:" with no additional text."""
        
        elif answer_type == "path":
            return """Please provide your answer in exactly this format:
ANSWER: [Entity1 → Entity2 → Entity3]

Examples:
ANSWER: [Project 10 → Launch 8 → Maintenance 4]
ANSWER: [Deadline 7 → Planning 12]

Use the arrow symbol (→) to separate entities in the path."""
        
        elif answer_type == "text":
            return """Please provide your answer in exactly this format:
ANSWER: [Your text answer]

Examples:
ANSWER: bidirectional conflict
ANSWER: inheritance conflict
ANSWER: temporal inconsistency

Provide a clear, concise text answer after "ANSWER:" with no additional explanation."""
        
        else:
            return """Please provide your answer in exactly this format:
ANSWER: [Your answer here]

Provide a clear, direct answer after "ANSWER:" based on the given context."""
    
    def validate_config(self) -> List[str]:
        """Validate the provider configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not self.config.model_name:
            errors.append("model_name is required")
        
        if self.config.max_tokens <= 0:
            errors.append("max_tokens must be positive")
        
        if not (0.0 <= self.config.temperature <= 2.0):
            errors.append("temperature must be between 0.0 and 2.0")
        
        if self.config.timeout <= 0:
            errors.append("timeout must be positive")
        
        return errors
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model and provider.
        
        Returns:
            Dictionary with model information
        """
        return {
            "provider": self.provider_name,
            "model": self.config.model_name,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "supports_batch": True,
            "supports_streaming": False,  # Override in subclasses if supported
        }
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test the connection to the provider.
        
        Returns:
            True if connection is successful, False otherwise
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model='{self.config.model_name}')"