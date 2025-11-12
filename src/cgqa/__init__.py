"""ChaosGraphQA (CGQA)

A comprehensive benchmark for testing reasoning capabilities of Large Language Models
using dynamically generated knowledge graphs.
"""

__version__ = "0.1.0"
__author__ = "ChaosGraphQA Contributors"

from .models.graph import Entity, KnowledgeGraph, Relationship
from .models.question import Answer, Question, QuestionType

# LLM integration (optional imports)
try:
    from .llm.evaluation.llm_evaluator import LLMEvaluator
    from .llm.evaluation.provider_factory import ProviderFactory

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMEvaluator = None  # type: ignore
    ProviderFactory = None  # type: ignore

__all__ = [
    "KnowledgeGraph",
    "Entity",
    "Relationship",
    "Question",
    "QuestionType",
    "Answer",
    "LLMEvaluator",
    "ProviderFactory",
    "LLM_AVAILABLE",
]
