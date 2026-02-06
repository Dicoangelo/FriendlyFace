"""Tests for the Merkle tree engine."""

import pytest

from friendlyface.core.merkle import MerkleTree


class TestMerkleTree:
    def test_empty_tree(self):
        tree = MerkleTree()
        assert tree.root is None
        assert tree.leaf_count == 0

    def test_single_leaf(self):
        tree = MerkleTree()
        tree.add_leaf("event_hash_1")
        assert tree.root is not None
        assert tree.leaf_count == 1

    def test_two_leaves_root(self):
        tree = MerkleTree()
        tree.add_leaf("a")
        tree.add_leaf("b")
        assert tree.leaf_count == 2
        root = tree.root
        assert root is not None

    def test_proof_single_leaf(self):
        tree = MerkleTree()
        tree.add_leaf("only_leaf")
        proof = tree.get_proof(0)
        assert proof.verify()
        assert tree.verify_proof(proof)

    def test_proof_two_leaves(self):
        tree = MerkleTree()
        tree.add_leaf("leaf_0")
        tree.add_leaf("leaf_1")

        proof0 = tree.get_proof(0)
        proof1 = tree.get_proof(1)

        assert proof0.verify()
        assert proof1.verify()
        assert tree.verify_proof(proof0)
        assert tree.verify_proof(proof1)

    def test_proof_four_leaves(self):
        tree = MerkleTree()
        for i in range(4):
            tree.add_leaf(f"leaf_{i}")

        for i in range(4):
            proof = tree.get_proof(i)
            assert proof.verify(), f"Proof failed for leaf {i}"
            assert tree.verify_proof(proof)

    def test_proof_odd_number_of_leaves(self):
        tree = MerkleTree()
        for i in range(5):
            tree.add_leaf(f"leaf_{i}")

        for i in range(5):
            proof = tree.get_proof(i)
            assert proof.verify(), f"Proof failed for leaf {i}"

    def test_proof_large_tree(self):
        """Test with 100+ leaves — forensic scenario."""
        tree = MerkleTree()
        for i in range(128):
            tree.add_leaf(f"event_hash_{i}")

        assert tree.leaf_count == 128

        # Verify random proofs
        for i in [0, 1, 50, 63, 64, 100, 127]:
            proof = tree.get_proof(i)
            assert proof.verify(), f"Proof failed for leaf {i}"

    def test_root_changes_on_append(self):
        tree = MerkleTree()
        tree.add_leaf("a")
        root1 = tree.root
        tree.add_leaf("b")
        root2 = tree.root
        assert root1 != root2

    def test_invalid_index_raises(self):
        tree = MerkleTree()
        tree.add_leaf("a")
        with pytest.raises(IndexError):
            tree.get_proof(1)
        with pytest.raises(IndexError):
            tree.get_proof(-1)

    def test_deterministic_root(self):
        """Same inputs must produce same root."""
        tree1 = MerkleTree()
        tree2 = MerkleTree()
        for x in ["a", "b", "c"]:
            tree1.add_leaf(x)
            tree2.add_leaf(x)
        assert tree1.root == tree2.root
