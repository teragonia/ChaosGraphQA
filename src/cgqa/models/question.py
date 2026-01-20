"""Question and answer data structures for ChaosGraphQA.

This module defines the question taxonomy and answer formats:
- QuestionType: Enum of reasoning types (multihop, temporal, etc.)
- AnswerType: Enum of answer formats (entity, list, boolean, etc.)
- Answer: Ground truth answer with confidence and explanation
- Question: Complete question with metadata and evaluation fields
- QuestionTemplate: Reusable template for question generation
- QuestionSet: Collection of questions for batch evaluation
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    """Reasoning types supported by the benchmark."""

    MULTIHOP = "multihop"  # Multi-hop path reasoning
    HIERARCHICAL = "hierarchical"  # Hierarchical/inheritance reasoning
    TEMPORAL = "temporal"  # Temporal/causal chain reasoning
    WEIGHTED = "weighted"  # Weighted path/optimization reasoning
    CONFLICTING = "conflicting"  # Conflict detection reasoning
    MIXED = "mixed"  # Mixed reasoning types


class AnswerType(str, Enum):
    """Expected answer formats for questions."""

    SINGLE_ENTITY = "single_entity"  # Single entity name
    ENTITY_LIST = "entity_list"  # List of entity names
    PATH = "path"  # Ordered sequence of entities
    BOOLEAN = "boolean"  # Yes/No answer
    NUMERIC = "numeric"  # Numeric value (count, length, weight)
    TEXT = "text"  # Free-form text


class Answer(BaseModel):
    """Ground truth answer with confidence and explanation.

    Stores the correct answer, its type, confidence score, and optionally
    a step-by-step explanation of how the answer was derived.
    """

    value: Union[str, List[str], bool, int, float, Dict[str, Any]]
    answer_type: AnswerType
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Answer confidence score"
    )
    explanation: Optional[str] = Field(
        default=None, description="Step-by-step reasoning for answer derivation"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional answer metadata"
    )


class Question(BaseModel):
    """Reasoning question with ground truth and evaluation fields.

    Represents a complete benchmark question including the text, expected answer,
    complexity level, and fields for storing LLM responses and scoring.
    """

    model_config = {"protected_namespaces": ()}

    id: str = Field(..., description="Unique question identifier")
    question_text: str = Field(..., description="Natural language question text")
    question_type: QuestionType = Field(..., description="Type of reasoning required")
    complexity_level: int = Field(
        ..., ge=1, le=4, description="Difficulty (1=easy to 4=hard)"
    )
    ground_truth: Answer = Field(..., description="Verified correct answer")
    context_entities: List[str] = Field(
        default_factory=list, description="Entity IDs referenced in question"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Template ID, generation params, etc."
    )

    # Evaluation results (populated after LLM evaluation)
    model_response: Optional[str] = Field(
        default=None, description="Raw LLM response text"
    )
    is_correct: Optional[bool] = Field(
        default=None, description="Binary correctness indicator"
    )
    score: Optional[float] = Field(
        default=None, description="Partial credit score (0.0 to 1.0)"
    )


class QuestionTemplate(BaseModel):
    """Template for generating questions of a specific type."""

    template: str = Field(..., description="Question template with placeholders")
    question_type: QuestionType
    complexity_level: int = Field(ge=1, le=4)
    required_graph_features: List[str] = Field(
        default_factory=list, description="Required graph properties"
    )
    variables: Dict[str, str] = Field(
        default_factory=dict, description="Variable descriptions"
    )
    answer_type: AnswerType

    def can_generate(self, graph_metadata: Dict[str, Any]) -> bool:
        """Check if this template can be used with the given graph."""
        for feature in self.required_graph_features:
            if feature not in graph_metadata:
                return False
        return True

    def format_question(self, variables: Dict[str, str]) -> str:
        """Fill in the template with actual values."""
        try:
            return self.template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing variable for template: {e}")


class QuestionSet(BaseModel):
    """A collection of questions for evaluation."""

    id: str = Field(..., description="Question set identifier")
    name: str = Field(..., description="Human-readable name")
    questions: List[Question] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_question(self, question: Question) -> None:
        """Add a question to the set."""
        self.questions.append(question)

    def get_by_type(self, question_type: QuestionType) -> List[Question]:
        """Get all questions of a specific type."""
        return [q for q in self.questions if q.question_type == question_type]

    def get_by_complexity(self, level: int) -> List[Question]:
        """Get all questions of a specific complexity level."""
        return [q for q in self.questions if q.complexity_level == level]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the question set."""
        if not self.questions:
            return {"total": 0}

        type_counts: dict = {}
        complexity_counts: dict = {}

        for q in self.questions:
            type_counts[q.question_type] = type_counts.get(q.question_type, 0) + 1
            complexity_counts[q.complexity_level] = (
                complexity_counts.get(q.complexity_level, 0) + 1
            )

        return {
            "total": len(self.questions),
            "by_type": type_counts,
            "by_complexity": complexity_counts,
            "avg_complexity": sum(q.complexity_level for q in self.questions)
            / len(self.questions),
        }
