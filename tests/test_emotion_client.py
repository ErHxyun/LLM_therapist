import json
import tempfile
import unittest
import wave
from pathlib import Path

from src.emotion.client import (
    EmotionSettings,
    EmotionSideChannel,
    NullEmotionSideChannel,
    _emotion_log_summary,
)


class EmotionClientTests(unittest.TestCase):
    def test_null_side_channel_is_noop(self):
        NullEmotionSideChannel().analyze_async(
            audio_file_path="/tmp/missing.wav",
            transcript="hello",
            sample_rate=16000,
            duration_seconds=1.0,
        )

    def test_analyze_posts_payload_and_records_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "input.wav"
            with wave.open(str(audio_path), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16000)
                writer.writeframes((0).to_bytes(2, byteorder="little", signed=True) * 1600)

            result_path = Path(tmpdir) / "emotion" / "results.jsonl"
            audio_dir = Path(tmpdir) / "emotion" / "audio"
            observed = {}

            def transport(url, payload, timeout_sec):
                observed["url"] = url
                observed["payload"] = payload
                observed["timeout_sec"] = timeout_sec
                observed["audio_exists_during_post"] = Path(payload["audio_file_path"]).exists()
                return {"final_result": {"risk_level": "low"}}

            client = EmotionSideChannel(
                EmotionSettings(
                    enabled=True,
                    service_url="http://127.0.0.1:8000/analyze",
                    user_id="8080",
                    language="en",
                    timeout_sec=3.0,
                    results_jsonl_path=str(result_path),
                    audio_dir=str(audio_dir),
                    keep_audio=False,
                ),
                transport=transport,
            )

            record = client.analyze(
                audio_file_path=str(audio_path),
                transcript="I feel okay.",
                sample_rate=16000,
                duration_seconds=0.1,
            )

            self.assertEqual(record["status"], "ok")
            self.assertEqual(observed["url"], "http://127.0.0.1:8000/analyze")
            self.assertEqual(observed["payload"]["transcript"], "I feel okay.")
            self.assertEqual(observed["payload"]["user_id"], "8080")
            self.assertEqual(observed["payload"]["sample_rate"], 16000)
            self.assertTrue(Path(observed["payload"]["audio_file_path"]).is_absolute())
            self.assertTrue(observed["audio_exists_during_post"])
            self.assertFalse(Path(observed["payload"]["audio_file_path"]).exists())

            lines = result_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            persisted = json.loads(lines[0])
            self.assertEqual(persisted["status"], "ok")
            self.assertEqual(persisted["response"]["final_result"]["risk_level"], "low")

    def test_emotion_log_summary_includes_demo_scores(self):
        summary = _emotion_log_summary(
            {
                "audio_emotion": [0, -1],
                "context_emotion": [0, 1],
                "audio_scores": {"arousal": 55, "tension": 45, "stability": 60},
                "text_scores": {"text_valence": 40, "certainty": 70},
                "emotion_comparison": {
                    "audio_vs_context_consistent": False,
                    "arousal_conflict": True,
                    "valence_conflict": False,
                    "contradiction_or_sarcasm": True,
                },
                "final_assessment": {"confidence": 80, "uncertainty": 20},
                "final_result": {"credibility_risk": 62, "risk_level": "Moderate"},
            }
        )

        self.assertEqual(summary["risk"], 62)
        self.assertEqual(summary["risk_level"], "Moderate")
        self.assertEqual(summary["confidence"], 80)
        self.assertEqual(summary["audio_emotion"], [0, -1])
        self.assertEqual(summary["context_emotion"], [0, 1])
        self.assertFalse(summary["consistent"])
        self.assertTrue(summary["arousal_conflict"])
        self.assertEqual(summary["audio_arousal"], 55)
        self.assertEqual(summary["text_valence"], 40)


if __name__ == "__main__":
    unittest.main()
