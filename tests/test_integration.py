"""Integration tests for the full ChaosGraphQA pipeline."""

import pytest

from cgqa.evaluators.ground_truth import GroundTruthVerifier
from cgqa.generators.multihop import MultiHopGenerator
from cgqa.models.question import QuestionType
from cgqa.questions.templates import QuestionGenerator


class TestFullPipeline:
    """Test the complete ChaosGraphQA pipeline."""

    def test_basic_pipeline(self):
        """Test the basic generate -> question -> verify pipeline."""
        # Generate knowledge graph
        generator = MultiHopGenerator(complexity_level=1, seed=42)
        kg = generator.generate()

        assert len(kg.entities) > 0
        assert len(kg.relationships) > 0

        # Generate questions
        question_gen = QuestionGenerator(seed=42)
        question_set = question_gen.generate_questions(
            kg,
            question_types=[QuestionType.MULTIHOP],
            num_questions_per_type=3,
            complexity_levels=[1],
        )

        # Should have generated some questions
        assert len(question_set.questions) > 0

        # All questions should be multi-hop type
        for question in question_set.questions:
            assert question.question_type == QuestionType.MULTIHOP
            assert question.complexity_level == 1
            assert question.ground_truth is not None

        # Verify ground truth
        verifier = GroundTruthVerifier(kg)

        # Test individual question verification
        if question_set.questions:
            question = question_set.questions[0]
            result = verifier.verify_question_answer(question)

            assert "is_valid" in result
            assert "confidence" in result
            assert "verification_method" in result
            assert isinstance(result["is_valid"], bool)
            assert 0.0 <= result["confidence"] <= 1.0

        # Test batch verification
        batch_results = verifier.verify_question_set(question_set.questions)

        assert "total_questions" in batch_results
        assert "valid_questions" in batch_results
        assert "validity_rate" in batch_results
        assert batch_results["total_questions"] == len(question_set.questions)
        assert 0.0 <= batch_results["validity_rate"] <= 1.0

    def test_different_complexity_levels(self):
        """Test pipeline with different complexity levels."""
        complexity_levels = [1, 2, 3]

        for complexity in complexity_levels:
            # Generate graph
            generator = MultiHopGenerator(complexity_level=complexity, seed=42)
            kg = generator.generate()

            # Generate questions
            question_gen = QuestionGenerator(seed=42)
            question_set = question_gen.generate_questions(
                kg,
                question_types=[QuestionType.MULTIHOP],
                num_questions_per_type=2,
                complexity_levels=[complexity],
            )

            # Should generate questions
            assert len(question_set.questions) > 0

            # Questions should match requested complexity
            for question in question_set.questions:
                assert question.complexity_level == complexity

            # Higher complexity should generally allow longer paths
            if complexity > 1:
                structured_paths = kg.metadata.get("structured_paths", [])
                if structured_paths:
                    max_path_length = max(path["length"] for path in structured_paths)
                    assert max_path_length >= complexity

    def test_reproducibility(self):
        """Test that the same seed produces reproducible results."""
        seed = 123

        # Generate first pipeline
        gen1 = MultiHopGenerator(complexity_level=2, seed=seed)
        kg1 = gen1.generate()

        qgen1 = QuestionGenerator(seed=seed)
        questions1 = qgen1.generate_questions(kg1, num_questions_per_type=3)

        # Generate second pipeline with same seed
        gen2 = MultiHopGenerator(complexity_level=2, seed=seed)
        kg2 = gen2.generate()

        qgen2 = QuestionGenerator(seed=seed)
        questions2 = qgen2.generate_questions(kg2, num_questions_per_type=3)

        # Should produce similar results
        assert len(kg1.entities) == len(kg2.entities)
        assert len(kg1.relationships) == len(kg2.relationships)
        assert len(questions1.questions) == len(questions2.questions)

    def test_metadata_propagation(self):
        """Test that metadata is properly propagated through pipeline."""
        # Generate graph
        generator = MultiHopGenerator(complexity_level=2, seed=42)
        kg = generator.generate()

        # Check graph metadata
        assert "generator_type" in kg.metadata
        assert "complexity_level" in kg.metadata
        assert "structured_paths" in kg.metadata

        # Generate questions
        question_gen = QuestionGenerator(seed=42)
        question_set = question_gen.generate_questions(kg)

        # Check question set metadata
        assert "graph_stats" in question_set.metadata
        assert "generation_params" in question_set.metadata

        # Check individual question metadata
        for question in question_set.questions:
            assert question.metadata is not None
            if "template_id" in question.metadata:
                assert isinstance(question.metadata["template_id"], str)

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Very small graph
        small_generator = MultiHopGenerator(complexity_level=1, seed=42)
        small_kg = small_generator.generate()

        # Should still work with small graphs
        question_gen = QuestionGenerator(seed=42)
        questions = question_gen.generate_questions(small_kg, num_questions_per_type=1)

        # Should handle gracefully (may generate fewer questions)
        assert isinstance(questions.questions, list)

        # Verify any generated questions
        if questions.questions:
            verifier = GroundTruthVerifier(small_kg)
            result = verifier.verify_question_answer(questions.questions[0])
            assert isinstance(result, dict)

    def test_question_diversity(self):
        """Test that different question templates produce diverse questions."""
        generator = MultiHopGenerator(complexity_level=3, seed=42)
        kg = generator.generate()

        question_gen = QuestionGenerator(seed=42)
        question_set = question_gen.generate_questions(
            kg,
            num_questions_per_type=10,  # Generate more questions
            complexity_levels=[2, 3],
        )

        if len(question_set.questions) >= 2:
            # Check that we get different question texts
            question_texts = [q.question_text for q in question_set.questions]
            unique_texts = set(question_texts)

            # Should have some diversity (not all identical)
            assert len(unique_texts) > 1

            # Check different answer types
            answer_types = [q.ground_truth.answer_type for q in question_set.questions]
            # May or may not have different types depending on templates
            assert all(answer_type is not None for answer_type in answer_types)
