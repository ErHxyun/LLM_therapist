import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NodeStatusDashboardTests(unittest.TestCase):
    def test_package_json_exposes_monitor_script(self):
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(package["scripts"]["monitor"], "node scripts/status_dashboard.js")
        self.assertTrue(package["private"])

    def test_status_dashboard_uses_python_monitor_as_read_only_source(self):
        script = (ROOT / "scripts" / "status_dashboard.js").read_text(encoding="utf-8")

        self.assertIn('const UPSTREAM_URL = process.env.CAITI_MONITOR_URL || "http://127.0.0.1:8765";', script)
        self.assertIn('url.pathname === "/api/status"', script)
        self.assertIn('url.pathname === "/events"', script)
        self.assertIn('url.pathname === "/health"', script)
        self.assertNotIn("Jetson.GPIO", script)
        self.assertNotIn("RPi.GPIO", script)


if __name__ == "__main__":
    unittest.main()
