"""Base LLM provider interface.

This module defines the unified interface for all LLM providers:
- LLMConfig: Configuration dataclass for provider settings
- LLMResponse: Standardized response format with token counts
- BaseLLMProvider: Abstract base class for provider implementations
- Prompt formatting for consistent question presentation
- Rate limiting and error handling
"""

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Module-level provider semaphores for concurrent request rate limiting
_PROVIDER_SEMAPHORES: Dict[str, threading.Semaphore] = {}
_SEMAPHORE_LOCK = threading.Lock()


class LLMConfig(BaseModel):
    """Provider configuration with API keys and generation parameters."""

    model_config = {"protected_namespaces": (), "extra": "allow"}

    provider_name: str = Field(..., description="Name of the LLM provider")
    model_name: str = Field(..., description="Specific model identifier")
    api_key: Optional[str] = Field(
        default=None, description="API key for authentication"
    )
    api_base: Optional[str] = Field(default=None, description="Custom API base URL")
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens to generate (None = use model default)",
    )
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="Sampling temperature"
    )
    timeout: int = Field(default=180, description="Request timeout in seconds")
    rate_limit_delay: float = Field(
        default=0.1, description="Delay between requests (seconds)"
    )
    max_concurrent_requests: int = Field(
        default=50, ge=1, description="Maximum concurrent requests for this provider"
    )

    extra_params: Dict[str, Any] = Field(
        default_factory=dict, description="Additional provider-specific parameters"
    )


@dataclass
class LLMResponse:
    """Standardized response from LLM providers."""

    text: str
    model_name: str
    provider_name: str
    timestamp: float = field(default_factory=time.time)

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    request_id: Optional[str] = None
    response_time: Optional[float] = None
    raw_response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider_name = config.provider_name
        self._setup_client()
        self._init_semaphore()

    def _init_semaphore(self) -> None:
        """Initialize per-provider semaphore for rate limiting concurrent requests."""
        global _PROVIDER_SEMAPHORES, _SEMAPHORE_LOCK

        with _SEMAPHORE_LOCK:
            if self.provider_name not in _PROVIDER_SEMAPHORES:
                _PROVIDER_SEMAPHORES[self.provider_name] = threading.Semaphore(
                    self.config.max_concurrent_requests
                )

    @abstractmethod
    def _setup_client(self) -> None:
        """Initialize the provider-specific client."""
        pass

    @abstractmethod
    def _make_request(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Make a request to the LLM provider.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for this request

        Returns:
            LLMResponse object with the result
        """
        pass

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate a response from the LLM with rate limiting.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (merged with config)

        Returns:
            LLMResponse object
        """
        start_time = time.time()

        # Acquire semaphore for rate limiting concurrent requests
        semaphore = _PROVIDER_SEMAPHORES.get(self.provider_name)

        try:
            # Use semaphore if available (concurrent mode)
            if semaphore:
                with semaphore:
                    return self._generate_internal(prompt, start_time, **kwargs)
            else:
                # No semaphore (shouldn't happen, but fallback)
                return self._generate_internal(prompt, start_time, **kwargs)

        except Exception as e:
            return LLMResponse(
                text="",
                model_name=self.config.model_name,
                provider_name=self.provider_name,
                error=str(e),
                error_type=type(e).__name__,
                response_time=time.time() - start_time,
            )

    def _generate_internal(
        self, prompt: str, start_time: float, **kwargs: Any
    ) -> LLMResponse:
        """Internal generate logic with rate limiting delay and request handling.

        Args:
            prompt: Input prompt
            start_time: Timestamp when generation started
            **kwargs: Additional parameters

        Returns:
            LLMResponse object
        """
        if self.config.rate_limit_delay > 0:
            time.sleep(self.config.rate_limit_delay)

        request_params = {
            "temperature": self.config.temperature,
            **self.config.extra_params,
            **kwargs,
        }

        # Only include max_tokens if it's explicitly set (not None)
        if self.config.max_tokens is not None:
            request_params["max_tokens"] = self.config.max_tokens

        response = self._make_request(prompt, **request_params)
        response.response_time = time.time() - start_time

        return response

    def batch_generate(self, prompts: List[str], **kwargs: Any) -> List[LLMResponse]:
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

    def format_question_prompt(
        self,
        question: str,
        context: Optional[str] = None,
        answer_type: str = "",
        question_type: Optional[str] = None,
    ) -> str:
        """Format a question into a prompt suitable for the LLM.

        Args:
            question: The question to ask
            context: Optional context (e.g., knowledge graph description)
            answer_type: Expected answer type for format guidance
            question_type: Type of question (e.g., "conflicting", "multihop")

        Returns:
            Formatted prompt string
        """
        format_instructions = self._get_format_instructions(
            answer_type, question_type, question
        )

        # Add special instructions for conflicting benchmarks
        conflicting_instructions = ""
        if question_type == "conflicting":
            conflicting_instructions = """
CRITICAL INSTRUCTIONS FOR REASONING ABOUT CONFLICTING INFORMATION:

When determining if entities are "connected through consistent (non-conflicting) relationships":

1. IDENTIFY CONTRADICTORY RELATIONSHIPS (not entities):
   Examine relationships for semantic contradictions:
   - Multiple mutually exclusive states: "X has_state Is Full" AND "X has_state Is Empty"
   - Contradictory type assignments: "X is_a Cat" AND "X is_a Dog" (if mutually exclusive)
   - Explicit contradictions: "X contradicts Y" means X and Y cannot coexist
   - Temporal impossibilities: cycles in temporal ordering

2. MARK ONLY THE SPECIFIC CONTRADICTORY RELATIONSHIPS AS INVALID:
   IMPORTANT: If entity A has contradictory relationships, only those SPECIFIC relationships are invalid.
   Other relationships involving A may still be valid!

   Example:
   - "CKaqu has_state Is Full" (INVALID - contradictory)
   - "CKaqu has_state Is Empty" (INVALID - contradictory)
   - "CKaqu connected_to CNid" (STILL VALID - not contradictory)

3. FIND PATHS USING ONLY VALID RELATIONSHIPS:
   Build paths between entities using relationships that are NOT marked as invalid.
   You CAN traverse through entities that have some contradictory relationships,
   as long as you don't use the contradictory relationships themselves.

4. ANSWER: Entities are "consistently connected" if there exists ANY path using exclusively valid relationships.

Key principle: Avoid contradictory RELATIONSHIPS, not entities that have contradictions.

"""

        base_prompt = f"""Context: {context}
{conflicting_instructions}
Question: {question}

{format_instructions}"""

        return base_prompt

    def _get_format_instructions(
        self,
        answer_type: str,
        question_type: Optional[str] = None,
        question: Optional[str] = None,
    ) -> str:
        """Get formatting instructions based on answer type and question type."""
        instructions = {
            "single_entity": """
═══════════════════════════════════════════════════════════════
CRITICAL: Your response MUST end with this EXACT format:
═══════════════════════════════════════════════════════════════

ANSWER: [Entity Name]

Examples of CORRECT format:
  ANSWER: Project 10
  ANSWER: Maintenance 4
  ANSWER: Launch 8

IMPORTANT:
• Start your final line with "ANSWER:" (all caps, with colon)
• Put only the entity name after "ANSWER:"
• Do NOT add any text after your answer
• You may explain your reasoning BEFORE the answer line""",
            "entity_list": """
═══════════════════════════════════════════════════════════════
CRITICAL: Your response MUST end with this EXACT format:
═══════════════════════════════════════════════════════════════

ANSWER: [Entity1, Entity2, Entity3]

Examples of CORRECT format:
  ANSWER: [Project 10, Launch 8]
  ANSWER: [Planning 12, Review 6, Maintenance 4]
  ANSWER: []

IMPORTANT:
• Start your final line with "ANSWER:" (all caps, with colon)
• Put entities inside square brackets: []
• Separate entities with commas
• Use empty brackets [] if there are no entities
• Do NOT add any text after your answer
• You may explain your reasoning BEFORE the answer line""",
            "boolean": """
═══════════════════════════════════════════════════════════════
CRITICAL: Your response MUST end with this EXACT format:
═══════════════════════════════════════════════════════════════

ANSWER: [Yes or No]

Examples of CORRECT format:
  ANSWER: Yes
  ANSWER: No

IMPORTANT:
• Start your final line with "ANSWER:" (all caps, with colon)
• Put only "Yes" or "No" after "ANSWER:"
• Do NOT write "ANSWER: Yes, because..." or any elaboration
• Do NOT add any text after your answer
• You may explain your reasoning BEFORE the answer line""",
            "numeric": """
═══════════════════════════════════════════════════════════════
CRITICAL: Your response MUST end with this EXACT format:
═══════════════════════════════════════════════════════════════

ANSWER: [Number]

Examples of CORRECT format:
  ANSWER: 3
  ANSWER: 7.5
  ANSWER: 0

IMPORTANT:
• Start your final line with "ANSWER:" (all caps, with colon)
• Put only the number after "ANSWER:"
• Do NOT write "ANSWER: The answer is 3" or "ANSWER: 3 steps"
• Do NOT add any text after your answer
• You may explain your reasoning BEFORE the answer line""",
            "path": """
═══════════════════════════════════════════════════════════════
CRITICAL: Your response MUST end with this EXACT format:
═══════════════════════════════════════════════════════════════

ANSWER: [Entity1 → Entity2 → Entity3]

Examples of CORRECT format:
  ANSWER: [Project 10 → Launch 8 → Maintenance 4]
  ANSWER: [Deadline 7 → Planning 12]

IMPORTANT:
• Start your final line with "ANSWER:" (all caps, with colon)
• Put the path inside square brackets: []
• Use arrow symbol → to separate entities
• Follow only the directional relationships from the context
• Do NOT add any text after your answer
• You may explain your reasoning BEFORE the answer line""",
            "text": """
═══════════════════════════════════════════════════════════════
CRITICAL: Your response MUST end with this EXACT format:
═══════════════════════════════════════════════════════════════

ANSWER: [Your text answer]

Examples of CORRECT format:
  ANSWER: bidirectional conflict
  ANSWER: inheritance conflict
  ANSWER: temporal inconsistency

IMPORTANT:
• Start your final line with "ANSWER:" (all caps, with colon)
• Put your concise text answer after "ANSWER:"
• Do NOT add any additional explanation after your answer
• You may explain your reasoning BEFORE the answer line""",
        }

        base_instruction = instructions.get(
            answer_type,
            """
═══════════════════════════════════════════════════════════════
CRITICAL: Your response MUST end with this EXACT format:
═══════════════════════════════════════════════════════════════

ANSWER: [Your answer here]

IMPORTANT:
• Start your final line with "ANSWER:" (all caps, with colon)
• Do NOT add any text after your answer
• You may explain your reasoning BEFORE the answer line""",
        )

        # Add conflict type information for conflicting questions with text answers
        if question_type == "conflicting" and answer_type == "text":
            conflict_types_info = """

Valid conflict types (choose one):
- direct_contradiction: Two relationships directly contradict each other about the same fact (e.g., X is_located_in Y vs X is_not_in Y)
- transitive_conflict: Transitive relationships lead to a logical contradiction (e.g., A > B > C > A, creating an impossible cycle)
- inheritance_conflict: Conflicting inheritance or classification (e.g., X is_a Cat AND X is_a Dog)
- temporal_conflict: Conflicting temporal relationships (e.g., A is_before B AND B is_before A)
- exclusivity_conflict: Mutually exclusive properties or states (e.g., X has_state Is Alive AND X has_state Is Dead)
- capacity_conflict: Impossible capacity or location constraints (e.g., entity in multiple exclusive locations simultaneously)"""
            base_instruction = base_instruction + conflict_types_info

        # Add clarification for "final outcome" causal chain questions
        if (
            question
            and "final outcome" in question.lower()
            and answer_type == "single_entity"
        ):
            final_outcome_info = """

Note: The final outcome is the entity at the end of the longest causal chain. If multiple paths exist, follow the chain with the most causal steps."""
            base_instruction = base_instruction + final_outcome_info

        return base_instruction

    def validate_config(self) -> List[str]:
        """Validate the provider configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if not self.config.model_name:
            errors.append("model_name is required")

        # Only validate max_tokens if it's explicitly set (not None)
        if self.config.max_tokens is not None and self.config.max_tokens <= 0:
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
            "supports_streaming": False,
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
