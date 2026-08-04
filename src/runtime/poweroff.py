from __future__ import annotations

from pathlib import Path

from src.utils import config_loader
from src.utils.log_util import get_logger

logger = get_logger("SystemPower")


def request_system_poweroff(reason: str, request_path: str | None = None) -> bool:
    raw_path = str(
        config_loader.SESSION_POWEROFF_REQUEST_PATH if request_path is None else request_path
    ).strip()
    if not raw_path:
        logger.warning("System poweroff request skipped for reason=%s because no marker path is configured.", reason)
        return False
    marker = Path(raw_path)
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{reason}\n", encoding="utf-8")
        logger.info("System poweroff marker created. reason=%s marker=%s", reason, marker)
        return True
    except Exception as exc:
        logger.warning("System poweroff marker creation failed for reason=%s marker=%s: %s", reason, marker, exc)
    return False


def clear_system_poweroff_request(request_path: str | None = None) -> None:
    raw_path = str(
        config_loader.SESSION_POWEROFF_REQUEST_PATH if request_path is None else request_path
    ).strip()
    if not raw_path:
        return
    marker = Path(raw_path)
    try:
        marker.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("Failed to clear stale system poweroff marker %s: %s", marker, exc)


__all__ = ["clear_system_poweroff_request", "request_system_poweroff"]
