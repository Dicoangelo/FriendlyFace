"""Custom exception hierarchy for FriendlyFace.

Provides structured error types that the centralized error middleware
translates into consistent JSON responses.
"""

from __future__ import annotations


class FriendlyFaceError(Exception):
    """Base exception for all FriendlyFace errors."""

    status_code: int = 500
    error_type: str = "internal_error"

    def __init__(self, message: str = "An internal error occurred") -> None:
        self.message = message
        super().__init__(message)


class StorageError(FriendlyFaceError):
    """Database or storage layer failure."""

    status_code = 503
    error_type = "storage_error"


class NotFoundError(FriendlyFaceError):
    """Requested resource was not found."""

    status_code = 404
    error_type = "not_found"


class ValidationError(FriendlyFaceError):
    """Input validation failure beyond Pydantic constraints."""

    status_code = 400
    error_type = "validation_error"


class CryptoError(FriendlyFaceError):
    """Cryptographic operation failure (DID, ZK, VC)."""

    status_code = 400
    error_type = "crypto_error"


class ChainIntegrityError(FriendlyFaceError):
    """Forensic hash chain integrity violation."""

    status_code = 422
    error_type = "chain_integrity_error"


class ConsentError(FriendlyFaceError):
    """Consent-related failure."""

    status_code = 403
    error_type = "consent_error"


class RateLimitError(FriendlyFaceError):
    """Rate limit exceeded."""

    status_code = 429
    error_type = "rate_limit_exceeded"
