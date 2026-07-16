import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.utils import io_record
from src.utils.io_record import HEADER, LONG_RESPONSE_PREFIX, segment_user_response


class IORecordSegmentationTest(unittest.TestCase):
    def test_splits_multi_dimension_contrast(self):
        self.assertEqual(
            segment_user_response(
                "I haven't slept well, but I've been eating regularly."
            ),
            ["I haven't slept well", "I've been eating regularly"],
        )

    def test_does_not_split_but_inside_a_word(self):
        self.assertEqual(
            segment_user_response("I ate butter and toast."),
            ["I ate butter and toast"],
        )

    def test_splits_case_insensitive_standalone_but(self):
        self.assertEqual(
            segment_user_response("I sleep poorly BUT I eat regularly"),
            ["I sleep poorly", "I eat regularly"],
        )

    def test_preserves_decimal_and_handles_sentence_punctuation(self):
        self.assertEqual(
            segment_user_response(
                "I sleep 7.5 hours! My appetite is okay? I exercise; not often."
            ),
            [
                "I sleep 7.5 hours",
                "My appetite is okay",
                "I exercise",
                "not often",
            ],
        )

    def test_long_response_prompt_is_persisted_as_internal_record_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record_path = str(Path(tmpdir) / "record.csv")
            pd.DataFrame(
                [["", 0, "", 1]],
                columns=HEADER,
            ).to_csv(record_path, index=False)

            original_record_csv = io_record.RECORD_CSV
            original_prefix = io_record._PENDING_QUESTION_PREFIX
            io_record.RECORD_CSV = record_path
            io_record._PENDING_QUESTION_PREFIX = ""
            try:
                io_record.set_question_prefix("I hear you.")
                io_record.log_question(
                    io_record.long_response_prompt("Can you tell me more?")
                )
                result = pd.read_csv(record_path)
            finally:
                io_record.RECORD_CSV = original_record_csv
                io_record._PENDING_QUESTION_PREFIX = original_prefix

        self.assertEqual(
            result.loc[0, "Question"],
            f"{LONG_RESPONSE_PREFIX}\nI hear you.\n\nCan you tell me more?",
        )


if __name__ == "__main__":
    unittest.main()
