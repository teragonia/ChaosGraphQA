"""LLM evaluation engine for KGRB."""

import time
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

from ...models.graph import KnowledgeGraph
from ...models.question import Question, QuestionSet
from ...questions.validators import AnswerValidator
from ..providers.base import BaseLLMProvider, LLMResponse
from .provider_factory import ProviderFactory


@dataclass 
class EvaluationResult:
    """Result of evaluating a single question."""
    
    question_id: str
    question_text: str
    question_type: str
    complexity_level: int
    
    # LLM response
    llm_response: str
    response_time: float
    
    # Ground truth
    ground_truth_answer: Any
    ground_truth_explanation: str
    
    # Evaluation metrics
    is_correct: bool
    score: float  # 0.0 to 1.0
    validation_explanation: str
    
    # Token usage
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    
    # Error information
    error: Optional[str] = None
    error_type: Optional[str] = None


@dataclass
class EvaluationSummary:
    """Summary of evaluation results."""
    
    model_name: str
    provider_name: str
    total_questions: int
    evaluation_time: float  # seconds
    timestamp: float = field(default_factory=time.time)
    
    # Overall metrics
    accuracy: float = 0.0  # Percentage of correct answers
    average_score: float = 0.0  # Average partial credit score
    
    # Performance by category
    by_question_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_complexity: Dict[int, Dict[str, float]] = field(default_factory=dict)
    
    # Token usage statistics
    total_tokens_used: int = 0
    average_tokens_per_question: float = 0.0
    
    # Error statistics
    error_rate: float = 0.0
    error_types: Dict[str, int] = field(default_factory=dict)
    
    # Individual results
    results: List[EvaluationResult] = field(default_factory=list)


class LLMEvaluator:
    """Main evaluation engine for testing LLMs on KGRB benchmarks."""
    
    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider
        self.validator = AnswerValidator()
    
    def evaluate_questions(
        self,
        questions: List[Question],
        kg: KnowledgeGraph,
        include_context: bool = True,
        show_progress: bool = True
    ) -> EvaluationSummary:
        """Evaluate a list of questions with the LLM.
        
        Args:
            questions: List of questions to evaluate
            kg: Knowledge graph containing the context
            include_context: Whether to include graph context in prompts
            show_progress: Whether to show progress information
            
        Returns:
            EvaluationSummary with all results
        """
        start_time = time.time()
        results = []
        
        if show_progress:
            try:
                from rich.progress import track
                question_iter = track(questions, description="Evaluating questions...")
            except ImportError:
                question_iter = questions
                if show_progress:
                    print(f"Evaluating {len(questions)} questions...")
        else:
            question_iter = questions
        
        for i, question in enumerate(question_iter):
            if show_progress and 'track' not in locals():
                print(f"Question {i+1}/{len(questions)}")
            
            result = self.evaluate_single_question(question, kg, include_context)
            results.append(result)
        
        # Create summary
        evaluation_time = time.time() - start_time
        summary = self._create_summary(results, evaluation_time)
        
        return summary
    
    def evaluate_single_question(
        self,
        question: Question,
        kg: KnowledgeGraph,
        include_context: bool = True
    ) -> EvaluationResult:
        """Evaluate a single question.
        
        Args:
            question: Question to evaluate
            kg: Knowledge graph for context
            include_context: Whether to include graph context
            
        Returns:
            EvaluationResult for this question
        """
        # Prepare context if requested
        context = None
        if include_context:
            context = self._prepare_context(question, kg)
        
        # Format prompt
        prompt = self.provider.format_question_prompt(question.question_text, context)
        
        # Get LLM response
        start_time = time.time()
        llm_response = self.provider.generate(prompt)
        response_time = time.time() - start_time
        
        # Validate answer
        if llm_response.error:
            # Handle LLM error
            return EvaluationResult(
                question_id=question.id,
                question_text=question.question_text,
                question_type=question.question_type,
                complexity_level=question.complexity_level,
                llm_response="",
                response_time=response_time,
                ground_truth_answer=question.ground_truth.value,
                ground_truth_explanation=question.ground_truth.explanation or "",
                is_correct=False,
                score=0.0,
                validation_explanation="LLM error occurred",
                error=llm_response.error,
                error_type=llm_response.error_type
            )
        
        # Validate the response
        validation_result = self.validator.validate_response(
            question, llm_response.text, strict=False
        )
        
        return EvaluationResult(
            question_id=question.id,
            question_text=question.question_text,
            question_type=question.question_type,
            complexity_level=question.complexity_level,
            llm_response=llm_response.text,
            response_time=response_time,
            ground_truth_answer=question.ground_truth.value,
            ground_truth_explanation=question.ground_truth.explanation or "",
            is_correct=validation_result["is_correct"],
            score=validation_result["score"],
            validation_explanation=validation_result["explanation"],
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            total_tokens=llm_response.total_tokens
        )
    
    def _prepare_context(self, question: Question, kg: KnowledgeGraph) -> str:
        """Prepare context from knowledge graph for a question.
        
        Args:
            question: The question being asked
            kg: Knowledge graph
            
        Returns:
            Formatted context string
        """
        # Get relevant entities
        context_entities = question.context_entities
        if not context_entities:
            # Fallback: use all entities (for small graphs)
            if len(kg.entities) <= 20:
                context_entities = list(kg.entities.keys())
            else:
                # For large graphs, this would need more sophisticated context selection
                context_entities = list(kg.entities.keys())[:20]
        
        # Build context description
        context_parts = []
        
        # Add entity information
        entities_described = set()
        for entity_id in context_entities:
            if entity_id in kg.entities and entity_id not in entities_described:
                entity = kg.entities[entity_id]
                context_parts.append(f"- {entity.name} ({entity.entity_type})")
                entities_described.add(entity_id)
        
        # Add relationships
        relevant_relationships = []
        for rel in kg.relationships:
            if rel.source in entities_described or rel.target in entities_described:
                relevant_relationships.append(rel)
        
        if relevant_relationships:
            context_parts.append("\nRelationships:")
            for rel in relevant_relationships[:50]:  # Limit to avoid token overflow
                source_entity = kg.entities.get(rel.source)
                target_entity = kg.entities.get(rel.target)
                
                if source_entity and target_entity:
                    context_parts.append(
                        f"- {source_entity.name} {rel.relation_type} {target_entity.name}"
                    )
        
        return "\n".join(context_parts)
    
    def _create_summary(self, results: List[EvaluationResult], evaluation_time: float) -> EvaluationSummary:
        """Create evaluation summary from individual results."""
        
        if not results:
            return EvaluationSummary(
                model_name=self.provider.config.model_name,
                provider_name=self.provider.provider_name,
                total_questions=0,
                evaluation_time=evaluation_time
            )
        
        # Overall metrics
        correct_count = sum(1 for r in results if r.is_correct)
        accuracy = correct_count / len(results) if results else 0.0
        average_score = sum(r.score for r in results) / len(results) if results else 0.0
        
        # Error statistics
        error_count = sum(1 for r in results if r.error is not None)
        error_rate = error_count / len(results) if results else 0.0
        error_types = {}
        for result in results:
            if result.error_type:
                error_types[result.error_type] = error_types.get(result.error_type, 0) + 1
        
        # Token usage
        total_tokens = sum(r.total_tokens or 0 for r in results)
        avg_tokens = total_tokens / len(results) if results else 0.0
        
        # Performance by question type
        by_question_type = {}
        type_groups = {}
        for result in results:
            q_type = result.question_type
            if q_type not in type_groups:
                type_groups[q_type] = []
            type_groups[q_type].append(result)
        
        for q_type, group_results in type_groups.items():
            type_correct = sum(1 for r in group_results if r.is_correct)
            type_accuracy = type_correct / len(group_results)
            type_avg_score = sum(r.score for r in group_results) / len(group_results)
            
            by_question_type[q_type] = {
                "accuracy": type_accuracy,
                "average_score": type_avg_score,
                "count": len(group_results)
            }
        
        # Performance by complexity
        by_complexity = {}
        complexity_groups = {}
        for result in results:
            complexity = result.complexity_level
            if complexity not in complexity_groups:
                complexity_groups[complexity] = []
            complexity_groups[complexity].append(result)
        
        for complexity, group_results in complexity_groups.items():
            comp_correct = sum(1 for r in group_results if r.is_correct)
            comp_accuracy = comp_correct / len(group_results)
            comp_avg_score = sum(r.score for r in group_results) / len(group_results)
            
            by_complexity[complexity] = {
                "accuracy": comp_accuracy,
                "average_score": comp_avg_score,
                "count": len(group_results)
            }
        
        return EvaluationSummary(
            model_name=self.provider.config.model_name,
            provider_name=self.provider.provider_name,
            total_questions=len(results),
            evaluation_time=evaluation_time,
            accuracy=accuracy,
            average_score=average_score,
            by_question_type=by_question_type,
            by_complexity=by_complexity,
            total_tokens_used=total_tokens,
            average_tokens_per_question=avg_tokens,
            error_rate=error_rate,
            error_types=error_types,
            results=results
        )
    
    def save_results(self, summary: EvaluationSummary, output_path: Union[str, Path]) -> None:
        """Save evaluation results to file.
        
        Args:
            summary: Evaluation summary to save
            output_path: Path to save results
        """
        output_path = Path(output_path)
        
        # Convert summary to serializable format
        results_data = {
            "metadata": {
                "model_name": summary.model_name,
                "provider_name": summary.provider_name,
                "total_questions": summary.total_questions,
                "evaluation_time": summary.evaluation_time,
                "timestamp": summary.timestamp,
            },
            "metrics": {
                "accuracy": summary.accuracy,
                "average_score": summary.average_score,
                "total_tokens_used": summary.total_tokens_used,
                "average_tokens_per_question": summary.average_tokens_per_question,
                "error_rate": summary.error_rate,
                "error_types": summary.error_types,
            },
            "performance": {
                "by_question_type": summary.by_question_type,
                "by_complexity": summary.by_complexity,
            },
            "individual_results": [
                {
                    "question_id": r.question_id,
                    "question_text": r.question_text,
                    "question_type": r.question_type,
                    "complexity_level": r.complexity_level,
                    "llm_response": r.llm_response,
                    "response_time": r.response_time,
                    "ground_truth_answer": r.ground_truth_answer,
                    "ground_truth_explanation": r.ground_truth_explanation,
                    "is_correct": r.is_correct,
                    "score": r.score,
                    "validation_explanation": r.validation_explanation,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "total_tokens": r.total_tokens,
                    "error": r.error,
                    "error_type": r.error_type,
                }
                for r in summary.results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
    
    @classmethod
    def from_model_string(cls, model_string: str, **kwargs) -> "LLMEvaluator":
        """Create evaluator from model string.
        
        Args:
            model_string: Model specification like "openai/gpt-4"
            **kwargs: Additional provider configuration
            
        Returns:
            LLMEvaluator instance
        """
        provider = ProviderFactory.create_provider(model_string, **kwargs)
        return cls(provider)