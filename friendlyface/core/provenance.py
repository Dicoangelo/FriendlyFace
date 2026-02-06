"""Provenance DAG for forensic data lineage tracking.

Implements Mohammed's ICDF2C 2024 provenance chain:
  training → model → inference → explanation → bundle

Each node is hash-sealed and references parent nodes, forming a
directed acyclic graph (DAG) of forensic evidence.
"""

from __future__ import annotations

from uuid import UUID

from friendlyface.core.models import ProvenanceNode, ProvenanceRelation


class ProvenanceDAG:
    """In-memory provenance DAG."""

    def __init__(self) -> None:
        self._nodes: dict[UUID, ProvenanceNode] = {}

    def add_node(
        self,
        entity_type: str,
        entity_id: str,
        parents: list[UUID] | None = None,
        relations: list[ProvenanceRelation] | None = None,
        metadata: dict | None = None,
    ) -> ProvenanceNode:
        """Add a provenance node to the DAG."""
        parent_ids = parents or []
        relation_types = relations or []

        # Validate parents exist
        for pid in parent_ids:
            if pid not in self._nodes:
                raise ValueError(f"Parent node {pid} not found in DAG")

        if parent_ids and len(relation_types) != len(parent_ids):
            raise ValueError("Must provide one relation per parent")

        node = ProvenanceNode(
            entity_type=entity_type,
            entity_id=entity_id,
            parents=parent_ids,
            relations=relation_types,
            metadata=metadata or {},
        ).seal()

        self._nodes[node.id] = node
        return node

    def get_node(self, node_id: UUID) -> ProvenanceNode | None:
        return self._nodes.get(node_id)

    def get_chain(self, node_id: UUID) -> list[ProvenanceNode]:
        """Walk the DAG upward from node_id, collecting the full provenance chain."""
        chain: list[ProvenanceNode] = []
        visited: set[UUID] = set()

        def _walk(nid: UUID) -> None:
            if nid in visited:
                return
            visited.add(nid)
            node = self._nodes.get(nid)
            if node is None:
                return
            for parent_id in node.parents:
                _walk(parent_id)
            chain.append(node)

        _walk(node_id)
        return chain

    def get_children(self, node_id: UUID) -> list[ProvenanceNode]:
        """Get all direct children of a node."""
        return [n for n in self._nodes.values() if node_id in n.parents]

    def verify_node(self, node_id: UUID) -> bool:
        """Verify hash integrity of a single node."""
        node = self._nodes.get(node_id)
        if node is None:
            return False
        return node.node_hash == node.compute_hash()

    def verify_chain(self, node_id: UUID) -> bool:
        """Verify hash integrity of the entire provenance chain."""
        chain = self.get_chain(node_id)
        return all(n.node_hash == n.compute_hash() for n in chain)

    @property
    def all_nodes(self) -> list[ProvenanceNode]:
        return list(self._nodes.values())
