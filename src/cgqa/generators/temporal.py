"""Temporal reasoning graph generator."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx

from ..models.graph import Entity, KnowledgeGraph, Relationship
from ..models.question import QuestionType
from .base_generator import BaseGraphGenerator


class TemporalGenerator(BaseGraphGenerator):
    """Generates graphs optimized for temporal reasoning tasks."""

    def __init__(
        self,
        complexity_level: int = 1,
        seed: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(complexity_level, seed, config)

        # Temporal specific configuration
        self.max_time_span = 365 * (1 + complexity_level)  # Days
        self.min_events = 5 + complexity_level * 3
        self.max_events = 10 + complexity_level * 5
        self.temporal_relations = self._get_temporal_relations()

        # Base time for generating events
        self.base_time = datetime(2020, 1, 1)  # Fixed reference point

    def _get_temporal_relations(self) -> List[Dict[str, Any]]:
        """Get different types of temporal relationships."""
        return [
            {
                "relation": "before",
                "inverse": "after",
                "description": "Event A happens before Event B",
            },
            {
                "relation": "during",
                "inverse": "contains",
                "description": "Event A happens during Event B",
            },
            {
                "relation": "starts",
                "inverse": "started_by",
                "description": "Event A starts Event B",
            },
            {
                "relation": "finishes",
                "inverse": "finished_by",
                "description": "Event A finishes Event B",
            },
            {
                "relation": "meets",
                "inverse": "met_by",
                "description": "Event A meets Event B (A ends when B starts)",
            },
            {
                "relation": "overlaps",
                "inverse": "overlapped_by",
                "description": "Event A overlaps with Event B",
            },
            {
                "relation": "causes",
                "inverse": "caused_by",
                "description": "Event A causes Event B",
            },
        ]

    def get_question_type(self) -> QuestionType:
        """Return the question type this generator supports."""
        return QuestionType.TEMPORAL

    def generate(self) -> KnowledgeGraph:
        """Generate a knowledge graph optimized for temporal reasoning."""

        # Create knowledge graph
        kg = KnowledgeGraph()

        # Generate events with timestamps
        events = self._create_temporal_events()

        # Add event entities to graph
        for event in events:
            kg.add_entity(event)

        # Create temporal relationships
        temporal_relationships = self._create_temporal_relationships(events)

        # Add relationships to graph
        for rel in temporal_relationships:
            kg.add_relationship(rel)

        # Add some non-event entities and relationships
        additional_entities = self._create_entities(
            self.rng.randint(3, 8),
            entity_types=["person", "place", "object"],
            prefix="A",
        )

        for entity in additional_entities.values():
            kg.add_entity(entity)

        # Connect additional entities to events
        self._connect_entities_to_events(
            kg, list(additional_entities.keys()), [e.id for e in events]
        )

        # Add metadata for question generation
        kg.metadata.update(
            {
                "generator_type": "temporal",
                "complexity_level": self.complexity_level,
                "time_span_days": self.max_time_span,
                "base_time": self.base_time.isoformat(),
                "events": [
                    {
                        "id": e.id,
                        "name": e.name,
                        "start_time": e.properties.get("start_time"),
                        "end_time": e.properties.get("end_time"),
                        "duration": e.properties.get("duration_days"),
                    }
                    for e in events
                ],
                "temporal_sequences": self._find_temporal_sequences(kg),
                "causal_chains": self._find_causal_chains(kg),
            }
        )

        return kg

    def _create_temporal_events(self) -> List[Entity]:
        """Create events with temporal properties."""

        event_types = [
            "meeting",
            "conference",
            "project",
            "training",
            "deadline",
            "launch",
            "review",
            "planning",
            "development",
            "testing",
            "presentation",
            "workshop",
            "milestone",
            "deployment",
            "maintenance",
        ]

        num_events = self.rng.randint(self.min_events, self.max_events)
        event_names = self.rng.sample(
            event_types * 3, min(num_events, len(event_types) * 3)
        )

        events = []
        current_time = self.base_time

        for i, event_name in enumerate(event_names):
            # Generate start time
            days_offset = self.rng.randint(0, self.max_time_span)
            start_time = self.base_time + timedelta(days=days_offset)

            # Generate duration (1 hour to 30 days)
            duration_hours = self.rng.choice(
                [
                    1,
                    2,
                    4,
                    8,  # Short events (hours)
                    24,
                    48,
                    72,  # Multi-day events
                    24 * 7,
                    24 * 14,
                    24 * 30,  # Long projects (weeks/months)
                ]
            )

            end_time = start_time + timedelta(hours=duration_hours)
            duration_days = duration_hours / 24

            entity_id = f"event_{event_name}_{i}_{self.rng.randint(1000, 9999)}"

            event = Entity(
                id=entity_id,
                name=f"{event_name.title()} {i+1}",
                entity_type="event",
                properties={
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_days": round(duration_days, 2),
                    "event_type": event_name,
                    "temporal_order": i,
                },
            )

            events.append(event)

        # Sort events by start time for easier processing
        events.sort(key=lambda e: e.properties["start_time"])

        # Update temporal order after sorting
        for i, event in enumerate(events):
            event.properties["temporal_order"] = i

        return events

    def _create_temporal_relationships(
        self, events: List[Entity]
    ) -> List[Relationship]:
        """Create temporal relationships between events."""
        relationships = []

        # Create sequential relationships (before/after)
        for i in range(len(events) - 1):
            curr_event = events[i]
            next_event = events[i + 1]

            # Basic before/after relationship
            if self.rng.random() < 0.7:  # 70% chance of explicit temporal link
                rel = Relationship(
                    source=curr_event.id,
                    target=next_event.id,
                    relation_type="before",
                    properties={"temporal_relationship": True},
                )
                relationships.append(rel)

        # Create overlapping relationships
        self._add_overlapping_relationships(events, relationships)

        # Create causal relationships
        self._add_causal_relationships(events, relationships)

        # Create containment relationships (during/contains)
        self._add_containment_relationships(events, relationships)

        return relationships

    def _add_overlapping_relationships(
        self, events: List[Entity], relationships: List[Relationship]
    ) -> None:
        """Add overlapping temporal relationships."""

        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                event_a = events[i]
                event_b = events[j]

                start_a = datetime.fromisoformat(event_a.properties["start_time"])
                end_a = datetime.fromisoformat(event_a.properties["end_time"])
                start_b = datetime.fromisoformat(event_b.properties["start_time"])
                end_b = datetime.fromisoformat(event_b.properties["end_time"])

                # Check for overlapping time periods
                if start_a < end_b and start_b < end_a:
                    # Events overlap
                    if self.rng.random() < 0.3:  # 30% chance to make it explicit
                        rel = Relationship(
                            source=event_a.id,
                            target=event_b.id,
                            relation_type="overlaps",
                            properties={"temporal_relationship": True},
                        )
                        relationships.append(rel)

    def _add_causal_relationships(
        self, events: List[Entity], relationships: List[Relationship]
    ) -> None:
        """Add causal relationships between events."""

        # Add some causal chains
        num_causal_chains = self.rng.randint(1, max(1, len(events) // 3))

        for _ in range(num_causal_chains):
            # Create a causal chain of 2-4 events
            chain_length = self.rng.randint(2, min(4, len(events)))
            chain_events = self.rng.sample(events, chain_length)

            # Sort by temporal order to maintain causality
            chain_events.sort(key=lambda e: e.properties["temporal_order"])

            for i in range(len(chain_events) - 1):
                cause_event = chain_events[i]
                effect_event = chain_events[i + 1]

                rel = Relationship(
                    source=cause_event.id,
                    target=effect_event.id,
                    relation_type="causes",
                    properties={
                        "temporal_relationship": True,
                        "causal_relationship": True,
                    },
                )
                relationships.append(rel)

    def _add_containment_relationships(
        self, events: List[Entity], relationships: List[Relationship]
    ) -> None:
        """Add containment relationships (during/contains)."""

        for i in range(len(events)):
            for j in range(len(events)):
                if i == j:
                    continue

                event_container = events[i]
                event_contained = events[j]

                start_container = datetime.fromisoformat(
                    event_container.properties["start_time"]
                )
                end_container = datetime.fromisoformat(
                    event_container.properties["end_time"]
                )
                start_contained = datetime.fromisoformat(
                    event_contained.properties["start_time"]
                )
                end_contained = datetime.fromisoformat(
                    event_contained.properties["end_time"]
                )

                # Check if contained event is fully within container event
                if (
                    start_container <= start_contained
                    and end_contained <= end_container
                    and event_container.properties["duration_days"]
                    > event_contained.properties["duration_days"]
                ):

                    if self.rng.random() < 0.2:  # 20% chance to make it explicit
                        rel = Relationship(
                            source=event_contained.id,
                            target=event_container.id,
                            relation_type="during",
                            properties={"temporal_relationship": True},
                        )
                        relationships.append(rel)

    def _connect_entities_to_events(
        self, kg: KnowledgeGraph, entity_ids: List[str], event_ids: List[str]
    ) -> None:
        """Connect non-event entities to events."""

        participation_relations = [
            "participates_in",
            "organizes",
            "attends",
            "hosts",
            "manages",
        ]

        for entity_id in entity_ids:
            # Connect to 1-3 events
            num_connections = self.rng.randint(1, min(3, len(event_ids)))
            connected_events = self.rng.sample(event_ids, num_connections)

            for event_id in connected_events:
                relation = self.rng.choice(participation_relations)

                rel = Relationship(
                    source=entity_id,
                    target=event_id,
                    relation_type=relation,
                    properties={"participation_relationship": True},
                )

                try:
                    kg.add_relationship(rel)
                except ValueError:
                    continue

    def _find_temporal_sequences(self, kg: KnowledgeGraph) -> List[Dict[str, Any]]:
        """Find temporal sequences for question generation."""
        sequences = []

        # Get all events sorted by time
        events = []
        for entity in kg.entities.values():
            if entity.entity_type == "event":
                events.append(entity)

        events.sort(key=lambda e: e.properties["start_time"])

        # Create sequences of consecutive events
        for start_idx in range(len(events)):
            for end_idx in range(start_idx + 2, min(start_idx + 5, len(events) + 1)):
                sequence = events[start_idx:end_idx]

                sequences.append(
                    {
                        "events": [e.id for e in sequence],
                        "event_names": [e.name for e in sequence],
                        "start_time": sequence[0].properties["start_time"],
                        "end_time": sequence[-1].properties["end_time"],
                        "length": len(sequence),
                    }
                )

        return sequences[:10]  # Limit to avoid too much metadata

    def _find_causal_chains(self, kg: KnowledgeGraph) -> List[Dict[str, Any]]:
        """Find causal chains for question generation."""
        chains = []

        # Build causal graph
        causal_graph: nx.DiGraph = nx.DiGraph()
        for rel in kg.relationships:
            if rel.relation_type == "causes":
                causal_graph.add_edge(rel.source, rel.target)

        # Find all paths in causal graph
        for source in causal_graph.nodes():
            for target in causal_graph.nodes():
                if source != target and nx.has_path(causal_graph, source, target):
                    try:
                        path = nx.shortest_path(causal_graph, source, target)
                        if len(path) > 1:  # Only non-trivial chains
                            chains.append(
                                {
                                    "cause": path[0],
                                    "effect": path[-1],
                                    "chain": path,
                                    "length": len(path) - 1,
                                }
                            )
                    except nx.NetworkXNoPath:
                        continue

        return chains[:10]  # Limit to avoid too much metadata
