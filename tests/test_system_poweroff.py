import tempfile
import unittest
from pathlib import Path

from src.runtime import poweroff


class SystemPoweroffTests(unittest.TestCase):
    def test_request_system_poweroff_writes_marker_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            marker = Path(tmpdir) / "caiti" / "poweroff-request"

            ok = poweroff.request_system_poweroff(
                "session complete",
                request_path=str(marker),
            )

            self.assertTrue(ok)
            self.assertEqual(marker.read_text(encoding="utf-8"), "session complete\n")

    def test_request_system_poweroff_returns_false_when_marker_path_is_empty(self):
        ok = poweroff.request_system_poweroff(
            "session complete",
            request_path="",
        )

        self.assertFalse(ok)

    def test_request_system_poweroff_returns_false_when_marker_write_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            marker = Path(tmpdir) / "already-a-directory"
            marker.mkdir()

            ok = poweroff.request_system_poweroff(
                "button shutdown",
                request_path=str(marker),
            )

            self.assertFalse(ok)

    def test_clear_system_poweroff_request_removes_marker_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            marker = Path(tmpdir) / "poweroff-request"
            marker.write_text("button shutdown\n", encoding="utf-8")

            poweroff.clear_system_poweroff_request(request_path=str(marker))

            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
