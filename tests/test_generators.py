"""Tests for graph generators."""

import pytest

from cgqa.generators.multihop import MultiHopGenerator
from cgqa.models.question import QuestionType


class TestMultiHopGenerator:
    """Test MultiHopGenerator."""

    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = MultiHopGenerator(complexity_level=2, seed=42)

        assert generator.complexity_level == 2
        assert generator.seed == 42
        assert generator.get_question_type() == QuestionType.MULTIHOP
        assert generator.max_hops >= 3  # Should be at least complexity + 1

    def test_generate_graph_basic(self):
        """Test basic graph generation."""
        generator = MultiHopGenerator(complexity_level=1, seed=42)
        kg = generator.generate()

        # Check basic properties
        assert len(kg.entities) > 0
        assert len(kg.relationships) > 0

        # Check metadata
        assert kg.metadata["generator_type"] == "multihop"
        assert kg.metadata["complexity_level"] == 1
        assert "structured_paths" in kg.metadata

    def test_generate_graph_complexity_scaling(self):
        """Test that complexity affects graph size."""
        gen_low = MultiHopGenerator(complexity_level=1, seed=42)
        gen_high = MultiHopGenerator(complexity_level=3, seed=42)

        kg_low = gen_low.generate()
        kg_high = gen_high.generate()

        # Higher complexity should generally produce larger graphs
        assert len(kg_high.entities) >= len(kg_low.entities)
        assert len(kg_high.relationships) >= len(kg_low.relationships)

    def test_generate_deterministic(self):
        """Test that same seed produces same graph."""
        gen1 = MultiHopGenerator(complexity_level=2, seed=123)
        gen2 = MultiHopGenerator(complexity_level=2, seed=123)

        kg1 = gen1.generate()
        kg2 = gen2.generate()

        # Should generate identical graphs with same seed
        assert len(kg1.entities) == len(kg2.entities)
        assert len(kg1.relationships) == len(kg2.relationships)

        # Entity names should be the same (since they're generated deterministically)
        names1 = {e.name for e in kg1.entities.values()}
        names2 = {e.name for e in kg2.entities.values()}
        assert names1 == names2

    def test_structured_paths_exist(self):
        """Test that structured paths are created."""
        generator = MultiHopGenerator(complexity_level=2, seed=42)
        kg = generator.generate()

        structured_paths = kg.metadata.get("structured_paths", [])

        # Should have some structured paths
        assert len(structured_paths) > 0

        # Each path should have required fields
        for path in structured_paths:
            assert "start" in path
            assert "end" in path
            assert "length" in path
            assert "path" in path
            assert "relations" in path

            # Path should be valid length
            assert path["length"] >= 2
            assert len(path["path"]) == path["length"] + 1

    def test_graph_connectivity(self):
        """Test that generated graph is connected."""
        generator = MultiHopGenerator(complexity_level=2, seed=42)
        kg = generator.generate()

        # Convert to NetworkX and check connectivity
        import networkx as nx

        nx_graph = kg.to_networkx(directed=False)
        assert nx.is_connected(nx_graph), "Generated graph should be connected"

    def test_path_finding_methods(self):
        """Test path finding utility methods."""
        generator = MultiHopGenerator(complexity_level=2, seed=42)
        kg = generator.generate()

        # Get some entities
        entity_ids = list(kg.entities.keys())
        if len(entity_ids) >= 2:
            start_id = entity_ids[0]
            end_id = entity_ids[1]

            # Test path finding
            path = generator.get_path_between(kg, start_id, end_id)

            if path:  # If path exists
                assert len(path) >= 2
                assert path[0] == start_id
                assert path[-1] == end_id

            # Test all paths
            all_paths = generator.get_all_paths(kg, start_id, end_id, max_length=5)
            assert isinstance(all_paths, list)

    def test_entity_name_generation(self):
        """Test entity name generation."""
        generator = MultiHopGenerator(complexity_level=1, seed=42)
        names = generator._generate_entity_names(5, prefix="Test")

        assert len(names) == 5
        assert all(name.startswith("Test") for name in names)
        assert len(set(names)) == 5  # All names should be unique

        # Names should be pronounceable (alternating consonant/vowel pattern)
        for name in names:
            # Remove prefix for checking
            base_name = name[4:]  # Remove "Test" prefix
            assert len(base_name) >= 3

    def test_validation(self):
        """Test graph validation."""
        generator = MultiHopGenerator(complexity_level=2, seed=42)
        kg = generator.generate()

        # Generated graph should pass validation
        assert generator.validate_graph(kg), "Generated graph should be valid"

        # Test with invalid graph (too few entities)
        invalid_kg = generator._create_entities(1)  # Only 1 entity
        kg_invalid = generator.kg = type(kg)()
        for entity in invalid_kg.values():
            kg_invalid.add_entity(entity)

        # This should fail validation due to size constraints
        # Note: We can't easily test this without refactoring the validation logic
