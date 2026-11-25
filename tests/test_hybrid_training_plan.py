import hashlib
import struct
import unittest

from tgat_phase1.hybrid_training_plan import _training_stats, _verify_paths


class HybridTrainingPlanTests(unittest.TestCase):
    def test_uses_outgoing_content_tokens_for_selected_row_input_support(self):
        docs = [
            {"lang": "eng_Latn", "text": "Hi"},
            {"lang": "jpn_Jpan", "text": "日本"},
            {"lang": "tur_Latn", "text": "İyi"},
        ]
        paths = [[1, 5, 2, 1], [2, 7], [7, 1, 7, 4, 2]]
        metrics, support = _training_stats(docs, paths, 3, {"42": [1, 2, 7], "43": [1, 2, 7]})
        self.assertEqual([r["causal_windows"] for r in metrics.values()], [2, 1, 3])
        self.assertEqual(metrics["jpn_Jpan"]["bytes"], 6)
        self.assertEqual(support["42"], support["43"])
        self.assertEqual(support["42"]["per_row_input_counts"], [2, 2, 2])
        self.assertEqual(support["42"]["input_tokens"], 8)
        self.assertEqual(support["42"]["selected_input_fraction"], 0.75)
        self.assertEqual(support["42"]["per_language"]["jpn_Jpan"]["per_row_input_counts"], [0, 1, 0])

    def test_fingerprint_checks_document_boundaries_and_token_order(self):
        pack = struct.pack("<5I", 2, 4, 5, 1, 6)
        expected = hashlib.sha256(pack).hexdigest()
        _verify_paths([[4, 5], [6]], expected, "test")
        with self.assertRaisesRegex(ValueError, "differ from the frozen G4 plan"):
            _verify_paths([[4], [5, 6]], expected, "test")


if __name__ == "__main__":
    unittest.main()
