"""Append-only forensic Merkle tree with inclusion proofs.

Based on BioZero pattern (arXiv:2409.17509) — designed for forensic
evidence integrity verification.

The tree is append-only: new leaves are added but existing ones never change.
This matches the immutable nature of forensic event chains.
"""

from __future__ import annotations

from uuid import uuid4

from friendlyface.core.models import MerkleNode, MerkleProof, sha256_hex


class MerkleTree:
    """In-memory append-only Merkle tree."""

    def __init__(self) -> None:
        self._leaves: list[str] = []
        self._nodes: dict[str, MerkleNode] = {}

    # --- Checkpoint persistence ---

    def to_checkpoint(self, event_index: dict | None = None) -> dict:
        """Serialize tree state for persistence."""
        return {
            "id": str(uuid4()),
            "leaf_count": len(self._leaves),
            "root_hash": self.root or "",
            "leaves": list(self._leaves),
            "event_index": {str(k): v for k, v in (event_index or {}).items()},
        }

    @classmethod
    def from_checkpoint(cls, data: dict) -> tuple["MerkleTree", dict]:
        """Restore tree from a checkpoint.

        Returns (tree, event_index) tuple.
        """
        tree = cls()
        for leaf_hash in data.get("leaves", []):
            tree._leaves.append(leaf_hash)
            node = MerkleNode(hash=leaf_hash, level=0, index=len(tree._leaves) - 1)
            tree._nodes[leaf_hash] = node

        # Restore event_index
        from uuid import UUID

        event_index = {}
        for k, v in data.get("event_index", {}).items():
            event_index[UUID(k)] = v

        return tree, event_index

    @property
    def leaf_count(self) -> int:
        return len(self._leaves)

    @property
    def root(self) -> str | None:
        if not self._leaves:
            return None
        return self._build_root()

    def add_leaf(self, data: str) -> str:
        """Add a leaf (event hash) to the tree. Returns leaf hash."""
        leaf_hash = sha256_hex(data)
        self._leaves.append(leaf_hash)
        node = MerkleNode(hash=leaf_hash, level=0, index=len(self._leaves) - 1)
        self._nodes[leaf_hash] = node
        return leaf_hash

    def get_proof(self, leaf_index: int) -> MerkleProof:
        """Generate an inclusion proof for the leaf at the given index."""
        if leaf_index < 0 or leaf_index >= len(self._leaves):
            raise IndexError(f"Leaf index {leaf_index} out of range [0, {len(self._leaves)})")

        proof_hashes: list[str] = []
        proof_directions: list[str] = []
        current_layer = list(self._leaves)
        idx = leaf_index

        while len(current_layer) > 1:
            # If odd number of nodes, duplicate the last
            if len(current_layer) % 2 == 1:
                current_layer.append(current_layer[-1])

            if idx % 2 == 0:
                # Sibling is to the right
                sibling = current_layer[idx + 1]
                proof_hashes.append(sibling)
                proof_directions.append("right")
            else:
                # Sibling is to the left
                sibling = current_layer[idx - 1]
                proof_hashes.append(sibling)
                proof_directions.append("left")

            # Build next layer
            next_layer = []
            for i in range(0, len(current_layer), 2):
                combined = sha256_hex(current_layer[i] + current_layer[i + 1])
                next_layer.append(combined)

            current_layer = next_layer
            idx = idx // 2

        root_hash = current_layer[0] if current_layer else self._leaves[0]

        return MerkleProof(
            leaf_hash=self._leaves[leaf_index],
            leaf_index=leaf_index,
            proof_hashes=proof_hashes,
            proof_directions=proof_directions,
            root_hash=root_hash,
        )

    def _build_root(self) -> str:
        """Compute the Merkle root from all current leaves."""
        if not self._leaves:
            return sha256_hex("")

        current_layer = list(self._leaves)

        while len(current_layer) > 1:
            if len(current_layer) % 2 == 1:
                current_layer.append(current_layer[-1])

            next_layer = []
            for i in range(0, len(current_layer), 2):
                combined = sha256_hex(current_layer[i] + current_layer[i + 1])
                next_layer.append(combined)

            current_layer = next_layer

        return current_layer[0]

    def verify_proof(self, proof: MerkleProof) -> bool:
        """Verify an inclusion proof against the current root."""
        return proof.verify() and proof.root_hash == self.root
