from types import SimpleNamespace
import unittest

from reproduction.check_zero_weight import compare_split


class ZeroWeightParityTests(unittest.TestCase):
    def test_counts_exact_ids_and_reports_first_difference(self):
        class Native:
            def encode(self, text, add_special_tokens):
                return SimpleNamespace(ids=[1, 2])

            def decode(self, ids, skip_special_tokens):
                return "text"

        class Portable:
            def __init__(self):
                self.paths = iter(((1, 2), (2, 1), (1, 2, 3)))

            def encode(self, text, rewards):
                return next(self.paths)

        lang = "eng_Latn"
        rows = [{"id": str(index), "lang": lang, "text": "text"} for index in range(3)]
        report = compare_split(rows, {lang: Native()}, {lang: Portable()}, {lang: {}})
        expected = {"checked": 3, "identical": 1, "mismatched": 2}
        self.assertEqual(report["by_language"]["all"], expected)
        self.assertEqual(report["by_language"][lang], expected)
        first, second = report["mismatches"]
        self.assertEqual(first["document_id"], "1")
        self.assertEqual(first["first_difference_index"], 0)
        self.assertEqual(first["g2_ids"], [1, 2])
        self.assertEqual(first["g3_zero_weight_ids"], [2, 1])
        self.assertEqual(second["first_difference_index"], 2)
        self.assertEqual(second["g2_tokens"], 2)
        self.assertEqual(second["g3_zero_weight_tokens"], 3)


if __name__ == "__main__":
    unittest.main()
