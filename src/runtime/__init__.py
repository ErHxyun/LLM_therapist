"""Runtime helpers for CaiTI process-local services."""

from src.runtime.user_context import (
    UserContext,
    activate_user_context,
    build_guest_user_id,
    build_user_context,
    get_current_user_context,
    normalize_spoken_user_id,
)

__all__ = [
    "UserContext",
    "activate_user_context",
    "build_guest_user_id",
    "build_user_context",
    "get_current_user_context",
    "normalize_spoken_user_id",
]
