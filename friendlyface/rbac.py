"""Role-Based Access Control for FriendlyFace (US-038).

Defines the role hierarchy and convenience utilities for endpoint
authorization.  The ``require_role`` dependency lives in ``auth.py``
and delegates to these definitions.

Roles (highest → lowest privilege):
    admin    — Full platform access, backup/restore, migrations, erasure
    analyst  — Train models, run simulations, trigger audits
    auditor  — Read compliance reports, view audit trails
    subject  — Data-subject self-service (consent, erasure requests)
    viewer   — Read-only access to non-sensitive endpoints
"""

from __future__ import annotations

from enum import StrEnum


class Role(StrEnum):
    """Enumerated platform roles."""

    ADMIN = "admin"
    ANALYST = "analyst"
    AUDITOR = "auditor"
    SUBJECT = "subject"
    VIEWER = "viewer"


#: Roles ordered from most to least privileged.
ROLE_HIERARCHY: list[Role] = [
    Role.ADMIN,
    Role.ANALYST,
    Role.AUDITOR,
    Role.SUBJECT,
    Role.VIEWER,
]

#: Mapping from each role to the set of roles it implicitly includes.
#: Admin inherits everything; analyst inherits auditor + viewer, etc.
ROLE_INCLUDES: dict[Role, frozenset[Role]] = {
    Role.ADMIN: frozenset(ROLE_HIERARCHY),
    Role.ANALYST: frozenset({Role.ANALYST, Role.AUDITOR, Role.VIEWER}),
    Role.AUDITOR: frozenset({Role.AUDITOR, Role.VIEWER}),
    Role.SUBJECT: frozenset({Role.SUBJECT, Role.VIEWER}),
    Role.VIEWER: frozenset({Role.VIEWER}),
}


def has_role(user_roles: list[str], required: str) -> bool:
    """Check if *user_roles* satisfy *required*, respecting hierarchy.

    Returns ``True`` when at least one of the caller's roles either
    matches *required* directly or includes it via the hierarchy.
    """
    for r in user_roles:
        try:
            role = Role(r)
        except ValueError:
            continue
        if required in ROLE_INCLUDES.get(role, frozenset()):
            return True
    return False
