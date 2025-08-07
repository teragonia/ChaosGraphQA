"""Question and answer data structures for ChaosGraphQA."""

from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    """Types of reasoning questions supported."""
    
    MULTIHOP = "multihop"
    HIERARCHICAL = "hierarchical" 
    TEMPORAL = "temporal"
    WEIGHTED = "weighted"
    CONFLICTING = "conflicting"
    MIXED = "mixed"


class AnswerType(str, Enum):
    """Types of expected answers."""
    
    SINGLE_ENTITY = "single_entity"
    ENTITY_LIST = "entity_list"
    PATH = "path"
    BOOLEAN = "boolean"
    NUMERIC = "numeric"
    TEXT = "text"


class Answer(BaseModel):
    """Represents a ground truth answer to a question."""
    
    value: Union[str, List[str], bool, float, Dict[str, Any]]
    answer_type: AnswerType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    explanation: Optional[str] = Field(default=None, description="Step-by-step reasoning")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Question(BaseModel):
    """Represents a reasoning question about a knowledge graph."""
    
    model_config = {"protected_namespaces": ()}
    
    id: str = Field(..., description="Unique question identifier")
    question_text: str = Field(..., description="The actual question")
    question_type: QuestionType = Field(..., description="Type of reasoning required")
    complexity_level: int = Field(..., ge=1, le=4, description="Difficulty level (1-4)")
    ground_truth: Answer = Field(..., description="Correct answer")
    context_entities: List[str] = Field(default_factory=list, description="Key entities mentioned in question")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional question metadata")
    
    # For evaluation
    model_response: Optional[str] = Field(default=None, description="LLM response to this question")
    is_correct: Optional[bool] = Field(default=None, description="Whether model got it right")
    score: Optional[float] = Field(default=None, description="Partial credit score")


class QuestionTemplate(BaseModel):
    """Template for generating questions of a specific type."""
    
    template: str = Field(..., description="Question template with placeholders")
    question_type: QuestionType
    complexity_level: int = Field(ge=1, le=4)
    required_graph_features: List[str] = Field(default_factory=list, description="Required graph properties")
    variables: Dict[str, str] = Field(default_factory=dict, description="Variable descriptions")
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
        
        type_counts = {}
        complexity_counts = {}
        
        for q in self.questions:
            type_counts[q.question_type] = type_counts.get(q.question_type, 0) + 1
            complexity_counts[q.complexity_level] = complexity_counts.get(q.complexity_level, 0) + 1
        
        return {
            "total": len(self.questions),
            "by_type": type_counts,
            "by_complexity": complexity_counts,
            "avg_complexity": sum(q.complexity_level for q in self.questions) / len(self.questions)
        }