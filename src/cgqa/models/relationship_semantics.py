"""Semantic directionality mapping for relationship types."""

from enum import Enum
from typing import Dict, Set, Tuple


class RelationDirection(Enum):
    """Direction semantics for relationships."""

    SOURCE_TO_TARGET = "source_to_target"  # A owns B means A → B
    TARGET_TO_SOURCE = "target_to_source"  # A created_by B means B → A
    BIDIRECTIONAL = "bidirectional"  # A colleague_of B means A ↔ B


class RelationshipSemantics:
    """Manages semantic directionality of relationships."""

    # Comprehensive mapping of relationship types to their semantic direction
    SEMANTIC_DIRECTIONS: Dict[str, RelationDirection] = {
        # Ownership and control (source → target)
        "owns": RelationDirection.SOURCE_TO_TARGET,
        "manages": RelationDirection.SOURCE_TO_TARGET,
        "leads": RelationDirection.SOURCE_TO_TARGET,
        "supervises": RelationDirection.SOURCE_TO_TARGET,
        "controls": RelationDirection.SOURCE_TO_TARGET,
        "reports_to": RelationDirection.SOURCE_TO_TARGET,
        "is_superior_to": RelationDirection.SOURCE_TO_TARGET,
        "is_subordinate_to": RelationDirection.SOURCE_TO_TARGET,
        # Creation and causation (target → source, because "A created_by B" means B created A)
        "created_by": RelationDirection.TARGET_TO_SOURCE,
        "caused_by": RelationDirection.TARGET_TO_SOURCE,
        "born_in": RelationDirection.TARGET_TO_SOURCE,  # "A born_in B" means B gave birth to/birthed A
        "started_by": RelationDirection.TARGET_TO_SOURCE,
        "finished_by": RelationDirection.TARGET_TO_SOURCE,
        "taught_by": RelationDirection.TARGET_TO_SOURCE,
        "managed_by": RelationDirection.TARGET_TO_SOURCE,
        "owned_by": RelationDirection.TARGET_TO_SOURCE,
        # Actions from source to target (source → target)
        "creates": RelationDirection.SOURCE_TO_TARGET,
        "causes": RelationDirection.SOURCE_TO_TARGET,
        "teaches": RelationDirection.SOURCE_TO_TARGET,
        "studies_at": RelationDirection.SOURCE_TO_TARGET,
        "works_at": RelationDirection.SOURCE_TO_TARGET,
        "lives_in": RelationDirection.SOURCE_TO_TARGET,
        "located_in": RelationDirection.SOURCE_TO_TARGET,
        "is_located_in": RelationDirection.SOURCE_TO_TARGET,
        "visits": RelationDirection.SOURCE_TO_TARGET,
        "attends": RelationDirection.SOURCE_TO_TARGET,
        "participates_in": RelationDirection.SOURCE_TO_TARGET,
        "organizes": RelationDirection.SOURCE_TO_TARGET,
        "hosts": RelationDirection.SOURCE_TO_TARGET,
        "uses": RelationDirection.SOURCE_TO_TARGET,
        "depends_on": RelationDirection.SOURCE_TO_TARGET,
        # Hierarchical relationships (source → target for containment)
        "part_of": RelationDirection.SOURCE_TO_TARGET,
        "member_of": RelationDirection.SOURCE_TO_TARGET,
        "contains": RelationDirection.SOURCE_TO_TARGET,
        "includes": RelationDirection.SOURCE_TO_TARGET,
        "is_a": RelationDirection.SOURCE_TO_TARGET,  # "A is_a B" means A is an instance of B
        "instance_of": RelationDirection.SOURCE_TO_TARGET,
        "subcategory": RelationDirection.SOURCE_TO_TARGET,
        # Symmetric/bidirectional relationships
        "colleague_of": RelationDirection.BIDIRECTIONAL,
        "friend_of": RelationDirection.BIDIRECTIONAL,
        "is_friend_of": RelationDirection.BIDIRECTIONAL,
        "connected_to": RelationDirection.BIDIRECTIONAL,
        "related_to": RelationDirection.BIDIRECTIONAL,
        "associated_with": RelationDirection.BIDIRECTIONAL,
        "similar_to": RelationDirection.BIDIRECTIONAL,
        "meets": RelationDirection.BIDIRECTIONAL,
        "knows": RelationDirection.BIDIRECTIONAL,
        "interacts_with": RelationDirection.BIDIRECTIONAL,
        "is_neutral_to": RelationDirection.BIDIRECTIONAL,
        # Temporal relationships (source → target)
        "before": RelationDirection.SOURCE_TO_TARGET,
        "after": RelationDirection.SOURCE_TO_TARGET,
        "during": RelationDirection.SOURCE_TO_TARGET,
        "is_before": RelationDirection.SOURCE_TO_TARGET,
        "is_after": RelationDirection.SOURCE_TO_TARGET,
        "is_simultaneous_with": RelationDirection.BIDIRECTIONAL,
        "starts": RelationDirection.SOURCE_TO_TARGET,
        "finishes": RelationDirection.SOURCE_TO_TARGET,
        "overlaps": RelationDirection.BIDIRECTIONAL,
        # Conflicting relationships (bidirectional opposition)
        "conflicts_with": RelationDirection.BIDIRECTIONAL,
        "contradicts": RelationDirection.BIDIRECTIONAL,
        "is_enemy_of": RelationDirection.BIDIRECTIONAL,
        "exclusive_with": RelationDirection.BIDIRECTIONAL,
        "mutually_exclusive": RelationDirection.BIDIRECTIONAL,
        # Negative relationships (usually source → target for action, bidirectional for state)
        "does_not_own": RelationDirection.SOURCE_TO_TARGET,
        "is_not_a": RelationDirection.SOURCE_TO_TARGET,
        "is_not_in": RelationDirection.SOURCE_TO_TARGET,
    }

    @classmethod
    def get_semantic_direction(cls, relation_type: str) -> RelationDirection:
        """Get the semantic direction for a relationship type."""
        return cls.SEMANTIC_DIRECTIONS.get(
            relation_type, RelationDirection.SOURCE_TO_TARGET
        )

    @classmethod
    def is_bidirectional(cls, relation_type: str) -> bool:
        """Check if a relationship type is bidirectional."""
        return (
            cls.get_semantic_direction(relation_type) == RelationDirection.BIDIRECTIONAL
        )

    @classmethod
    def get_effective_direction(
        cls, source: str, target: str, relation_type: str
    ) -> Tuple[str, str]:
        """Get the effective direction based on semantic meaning.

        Returns:
            Tuple of (from_entity, to_entity) representing the actual direction of the relationship
        """
        direction = cls.get_semantic_direction(relation_type)

        if direction == RelationDirection.SOURCE_TO_TARGET:
            return (source, target)
        elif direction == RelationDirection.TARGET_TO_SOURCE:
            return (target, source)
        else:  # BIDIRECTIONAL
            return (source, target)  # Keep original for bidirectional

    @classmethod
    def can_traverse(
        cls,
        from_entity: str,
        to_entity: str,
        relation_source: str,
        relation_target: str,
        relation_type: str,
    ) -> bool:
        """Check if we can traverse from from_entity to to_entity via this relationship.

        Args:
            from_entity: Entity we want to traverse from
            to_entity: Entity we want to traverse to
            relation_source: Source entity in the relationship record
            relation_target: Target entity in the relationship record
            relation_type: Type of relationship
        """
        direction = cls.get_semantic_direction(relation_type)

        if direction == RelationDirection.BIDIRECTIONAL:
            # Can traverse in both directions
            return (
                from_entity == relation_source and to_entity == relation_target
            ) or (from_entity == relation_target and to_entity == relation_source)

        # Get effective direction
        effective_from, effective_to = cls.get_effective_direction(
            relation_source, relation_target, relation_type
        )

        # Can only traverse in the semantic direction
        return from_entity == effective_from and to_entity == effective_to

    @classmethod
    def get_direction_explanation(cls, relation_type: str) -> str:
        """Get a human-readable explanation of the relationship direction."""
        direction = cls.get_semantic_direction(relation_type)

        if direction == RelationDirection.SOURCE_TO_TARGET:
            return f"'A {relation_type} B' means the relationship goes from A to B"
        elif direction == RelationDirection.TARGET_TO_SOURCE:
            return f"'A {relation_type} B' means the relationship goes from B to A (B performed the action on A)"
        else:  # BIDIRECTIONAL
            return f"'A {relation_type} B' means the relationship goes in both directions (A ↔ B)"

    @classmethod
    def generate_directionality_prompt_section(cls, relation_types: Set[str]) -> str:
        """Generate the directionality explanation section for prompts."""
        lines = [
            "Relationship Directionality:",
            "Based on semantic meaning, relationships have the following directions:",
        ]

        # Group by direction type
        source_to_target = []
        target_to_source = []
        bidirectional = []

        for rel_type in sorted(relation_types):
            direction = cls.get_semantic_direction(rel_type)
            if direction == RelationDirection.SOURCE_TO_TARGET:
                source_to_target.append(rel_type)
            elif direction == RelationDirection.TARGET_TO_SOURCE:
                target_to_source.append(rel_type)
            else:
                bidirectional.append(rel_type)

        if source_to_target:
            lines.append("")
            lines.append("Forward direction (A → B):")
            for rel_type in source_to_target:
                lines.append(f"  - 'A {rel_type} B' means A → B")

        if target_to_source:
            lines.append("")
            lines.append("Reverse direction (B → A):")
            for rel_type in target_to_source:
                lines.append(
                    f"  - 'A {rel_type} B' means B → A (B performed action on/created A)"
                )

        if bidirectional:
            lines.append("")
            lines.append("Bidirectional (A ↔ B):")
            for rel_type in bidirectional:
                lines.append(f"  - 'A {rel_type} B' means A ↔ B")

        lines.extend(
            [
                "",
                "For path-finding: only follow relationships in their semantic direction",
                "To traverse from B to A when relationship is A → B, there must be a separate relationship allowing B → A",
            ]
        )

        return "\n".join(lines)
