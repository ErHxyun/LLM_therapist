import unittest

from src.utils.io_record import segment_user_response


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


if __name__ == "__main__":
    unittest.main()
