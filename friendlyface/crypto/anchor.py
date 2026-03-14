"""Blockchain Merkle root anchoring (US-008).

Publishes Merkle root hashes to blockchain networks for external proof.
Supports Polygon, Base L2, and a NullAnchor no-op for local/demo deployments.

web3 is an OPTIONAL dependency — only required for PolygonAnchor and BaseAnchor.
NullAnchor works without any extra packages.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass


@dataclass
class AnchorResult:
    """Result of anchoring a Merkle root to a blockchain."""

    tx_hash: str
    block_number: int
    chain: str
    timestamp: float
    merkle_root: str

    def to_dict(self) -> dict:
        return asdict(self)


class BlockchainAnchor(ABC):
    """Abstract base class for blockchain anchoring."""

    @abstractmethod
    async def anchor_root(self, merkle_root: str) -> AnchorResult:
        """Publish a Merkle root hash to the blockchain.

        Args:
            merkle_root: The hex-encoded Merkle root hash to anchor.

        Returns:
            AnchorResult with transaction details.
        """


class NullAnchor(BlockchainAnchor):
    """No-op anchor for local/demo deployments. Always succeeds."""

    async def anchor_root(self, merkle_root: str) -> AnchorResult:
        return AnchorResult(
            tx_hash="0x" + "0" * 64,
            block_number=0,
            chain="none",
            timestamp=time.time(),
            merkle_root=merkle_root,
        )


class PolygonAnchor(BlockchainAnchor):
    """Publishes Merkle root hash to a Polygon smart contract.

    Requires the ``web3`` package: ``pip install web3``.
    """

    # Polygon mainnet RPC (public)
    DEFAULT_RPC = "https://polygon-rpc.com"

    # Minimal ABI for a store(bytes32) function
    STORE_ABI = [
        {
            "inputs": [{"internalType": "bytes32", "name": "root", "type": "bytes32"}],
            "name": "store",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function",
        }
    ]

    def __init__(
        self,
        private_key: str,
        contract_address: str = "0x0000000000000000000000000000000000000000",
        rpc_url: str | None = None,
    ) -> None:
        try:
            from web3 import Web3  # noqa: F401
        except ImportError:
            msg = (
                "web3 is required for PolygonAnchor. "
                "Install it with: pip install web3"
            )
            raise ImportError(msg)  # noqa: B904

        self._private_key = private_key
        self._contract_address = contract_address
        self._rpc_url = rpc_url or self.DEFAULT_RPC

    async def anchor_root(self, merkle_root: str) -> AnchorResult:
        from web3 import Web3

        w3 = Web3(Web3.HTTPProvider(self._rpc_url))
        account = w3.eth.account.from_key(self._private_key)

        contract = w3.eth.contract(
            address=Web3.to_checksum_address(self._contract_address),
            abi=self.STORE_ABI,
        )

        root_bytes = bytes.fromhex(merkle_root.removeprefix("0x").ljust(64, "0")[:64])

        tx = contract.functions.store(root_bytes).build_transaction(
            {
                "from": account.address,
                "nonce": w3.eth.get_transaction_count(account.address),
                "gas": 100_000,
                "gasPrice": w3.eth.gas_price,
            }
        )

        signed = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

        return AnchorResult(
            tx_hash=receipt.transactionHash.hex(),
            block_number=receipt.blockNumber,
            chain="polygon",
            timestamp=time.time(),
            merkle_root=merkle_root,
        )


class BaseAnchor(BlockchainAnchor):
    """Publishes Merkle root hash to Base L2.

    Requires the ``web3`` package: ``pip install web3``.
    """

    # Base mainnet RPC (public)
    DEFAULT_RPC = "https://mainnet.base.org"

    # Same minimal ABI
    STORE_ABI = PolygonAnchor.STORE_ABI

    def __init__(
        self,
        private_key: str,
        contract_address: str = "0x0000000000000000000000000000000000000000",
        rpc_url: str | None = None,
    ) -> None:
        try:
            from web3 import Web3  # noqa: F401
        except ImportError:
            msg = (
                "web3 is required for BaseAnchor. "
                "Install it with: pip install web3"
            )
            raise ImportError(msg)  # noqa: B904

        self._private_key = private_key
        self._contract_address = contract_address
        self._rpc_url = rpc_url or self.DEFAULT_RPC

    async def anchor_root(self, merkle_root: str) -> AnchorResult:
        from web3 import Web3

        w3 = Web3(Web3.HTTPProvider(self._rpc_url))
        account = w3.eth.account.from_key(self._private_key)

        contract = w3.eth.contract(
            address=Web3.to_checksum_address(self._contract_address),
            abi=self.STORE_ABI,
        )

        root_bytes = bytes.fromhex(merkle_root.removeprefix("0x").ljust(64, "0")[:64])

        tx = contract.functions.store(root_bytes).build_transaction(
            {
                "from": account.address,
                "nonce": w3.eth.get_transaction_count(account.address),
                "gas": 100_000,
                "gasPrice": w3.eth.gas_price,
            }
        )

        signed = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

        return AnchorResult(
            tx_hash=receipt.transactionHash.hex(),
            block_number=receipt.blockNumber,
            chain="base",
            timestamp=time.time(),
            merkle_root=merkle_root,
        )


def get_anchor(chain: str, private_key: str | None = None) -> BlockchainAnchor:
    """Factory: return the appropriate anchor implementation.

    Args:
        chain: One of 'none', 'polygon', 'base'.
        private_key: Blockchain wallet private key (required for polygon/base).

    Returns:
        A BlockchainAnchor instance.
    """
    chain = chain.lower().strip()
    if chain == "none" or chain == "":
        return NullAnchor()
    if chain == "polygon":
        if not private_key:
            msg = "FF_ANCHOR_KEY is required for Polygon anchoring"
            raise ValueError(msg)
        return PolygonAnchor(private_key=private_key)
    if chain == "base":
        if not private_key:
            msg = "FF_ANCHOR_KEY is required for Base anchoring"
            raise ValueError(msg)
        return BaseAnchor(private_key=private_key)
    msg = f"Unknown anchor chain: '{chain}'. Use 'none', 'polygon', or 'base'."
    raise ValueError(msg)
