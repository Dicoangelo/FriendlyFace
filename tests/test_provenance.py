"""Tests for the provenance DAG."""

import pytest

from friendlyface.core.models import ProvenanceRelation
from friendlyface.core.provenance import ProvenanceDAG


class TestProvenanceDAG:
    def test_add_root_node(self):
        dag = ProvenanceDAG()
        node = dag.add_node("dataset", "training_data_v1")
        assert node.node_hash != ""
        assert node.parents == []

    def test_linear_chain(self):
        """training → model → inference → explanation (Mohammed schema)."""
        dag = ProvenanceDAG()

        dataset = dag.add_node("dataset", "faces_v1")
        model = dag.add_node(
            "model",
            "resnet50-v1",
            parents=[dataset.id],
            relations=[ProvenanceRelation.DERIVED_FROM],
        )
        inference = dag.add_node(
            "inference",
            "inf_001",
            parents=[model.id],
            relations=[ProvenanceRelation.GENERATED_BY],
        )
        explanation = dag.add_node(
            "explanation",
            "exp_001",
            parents=[inference.id],
            relations=[ProvenanceRelation.DERIVED_FROM],
        )

        chain = dag.get_chain(explanation.id)
        assert len(chain) == 4
        assert chain[0].entity_type == "dataset"
        assert chain[-1].entity_type == "explanation"

    def test_invalid_parent_raises(self):
        from uuid import uuid4

        dag = ProvenanceDAG()
        with pytest.raises(ValueError, match="not found"):
            dag.add_node("model", "bad", parents=[uuid4()], relations=[ProvenanceRelation.USED])

    def test_mismatched_relations_raises(self):
        dag = ProvenanceDAG()
        root = dag.add_node("dataset", "data")
        with pytest.raises(ValueError, match="one relation per parent"):
            dag.add_node("model", "m1", parents=[root.id], relations=[])

    def test_verify_chain(self):
        dag = ProvenanceDAG()
        d = dag.add_node("dataset", "d1")
        m = dag.add_node("model", "m1", parents=[d.id], relations=[ProvenanceRelation.DERIVED_FROM])
        assert dag.verify_chain(m.id)

    def test_tamper_detection(self):
        dag = ProvenanceDAG()
        node = dag.add_node("dataset", "d1")
        # Tamper
        node.entity_id = "tampered"
        assert not dag.verify_node(node.id)

    def test_get_children(self):
        dag = ProvenanceDAG()
        root = dag.add_node("dataset", "d1")
        dag.add_node("model", "m1", parents=[root.id], relations=[ProvenanceRelation.DERIVED_FROM])
        dag.add_node("model", "m2", parents=[root.id], relations=[ProvenanceRelation.DERIVED_FROM])
        children = dag.get_children(root.id)
        assert len(children) == 2

    def test_dag_with_multiple_parents(self):
        dag = ProvenanceDAG()
        d1 = dag.add_node("dataset", "faces")
        d2 = dag.add_node("dataset", "voices")
        fused = dag.add_node(
            "model",
            "multimodal_v1",
            parents=[d1.id, d2.id],
            relations=[ProvenanceRelation.DERIVED_FROM, ProvenanceRelation.DERIVED_FROM],
        )
        chain = dag.get_chain(fused.id)
        assert len(chain) == 3  # d1, d2, fused
