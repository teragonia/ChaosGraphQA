"""Knowledge Graph Reasoning Benchmark (KGRB)

A comprehensive benchmark for testing reasoning capabilities of Large Language Models
using dynamically generated knowledge graphs.
"""

__version__ = "0.1.0"
__author__ = "KGRB Contributors"

from .models.graph import KnowledgeGraph, Entity, Relationship
from .models.question import Question, QuestionType, Answer

# LLM integration (optional imports)
try:
    from .llm.evaluation.llm_evaluator import LLMEvaluator
    from .llm.evaluation.provider_factory import ProviderFactory
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMEvaluator = None
    ProviderFactory = None

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