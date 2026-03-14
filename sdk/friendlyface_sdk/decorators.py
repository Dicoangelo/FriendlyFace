"""Decorators for forensic tracing."""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

from friendlyface_sdk.client import FriendlyFaceClient

F = TypeVar("F", bound=Callable[..., Any])


def forensic_trace(client: FriendlyFaceClient) -> Callable[[F], F]:
    """Decorator that logs function inputs/outputs as forensic events.

    Usage::

        @forensic_trace(client)
        def my_function(x, y):
            return x + y
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Log input event
            client.log_event(
                event_type="forensic_trace_input",
                content={
                    "actor": "sdk",
                    "function": func.__qualname__,
                    "args": repr(args),
                    "kwargs": repr(kwargs),
                },
            )
            result = func(*args, **kwargs)
            # Log output event
            client.log_event(
                event_type="forensic_trace_output",
                content={
                    "actor": "sdk",
                    "function": func.__qualname__,
                    "result": repr(result),
                },
            )
            return result

        return wrapper  # type: ignore[return-value]

    return decorator
