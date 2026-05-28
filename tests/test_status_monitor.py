import json
import urllib.request
import unittest

from src.runtime.status_monitor import StatusMonitor, StatusMonitorSettings


class StatusMonitorTests(unittest.TestCase):
    def test_snapshot_tracks_phase_lights_and_button(self):
        monitor = StatusMonitor(StatusMonitorSettings(enabled=False))

        monitor.set_phase("screening")
        monitor.set_light("blue", True)
        monitor.set_button_event("skip_to_cbt")
        snapshot = monitor.snapshot()

        self.assertEqual(snapshot["phase"], "screening")
        self.assertTrue(snapshot["lights"]["blue"])
        self.assertEqual(snapshot["button"]["last_event"], "skip_to_cbt")
        self.assertGreater(snapshot["version"], 0)

    def test_status_endpoint_returns_json(self):
        monitor = StatusMonitor(StatusMonitorSettings(enabled=True, host="127.0.0.1", port=0))
        try:
            self.assertTrue(monitor.start())
            monitor.set_phase("cbt")
            monitor.set_light("green", True)

            with urllib.request.urlopen(monitor.url + "/status", timeout=2) as response:
                data = json.loads(response.read().decode("utf-8"))
        finally:
            monitor.stop()

        self.assertEqual(data["phase"], "cbt")
        self.assertTrue(data["lights"]["green"])


if __name__ == "__main__":
    unittest.main()
