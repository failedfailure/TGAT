import math
import unittest

from tgat_phase1.hybrid_training_metrics import BASELINE, CELLS, INFERENCE_ONLY, TRAINED, hybrid_training_summary


def examples():
    sizes = (10, 30, 60, 20)
    langs = ("eng_Latn", "jpn_Jpan", "tur_Latn", "eng_Latn")
    bits = ((10, 30, 60, 20), (15, 27, 72, 20), (12, 30, 64, 18))
    return {
        cell: [{"id": i, "split": "test" if i < 3 else "validation", "lang": lang,
                "bytes": str(size), "tokens": size // (3 if cell == BASELINE else 2),
                "logp": -cost * math.log(2)}
               for i, (lang, size, cost) in enumerate(zip(langs, sizes, costs, strict=True))]
        for cell, costs in zip(CELLS, bits, strict=True)
    }


class HybridTrainingMetricsTests(unittest.TestCase):
    def test_weighted_bpb_and_positive_gap_recovery(self):
        cells = examples()
        cells[TRAINED].reverse()
        summary = {(row["split"], row["lang"]): row for row in hybrid_training_summary(cells)}
        test = summary["test", "all"]
        self.assertEqual((test["documents"], test["bytes"]), (3, 100))
        for field, expected in {
            "g1_bpe_bpb": 1.0, "g4_hybrid_bpb": 1.14, "g4_hybrid_trained_bpb": 1.06,
            "inference_gap_bpb": 0.14, "trained_gap_bpb": 0.06, "trained_vs_inference_bpb": -0.08,
            "recovery_fraction": 4 / 7, "g4_hybrid_tokens_per_byte": 0.5, "g4_hybrid_trained_tokens_per_byte": 0.5,
        }.items():
            with self.subTest(field=field):
                self.assertAlmostEqual(test[field], expected)
        self.assertAlmostEqual(summary["test", "eng_Latn"]["recovery_fraction"], 0.6)
        self.assertAlmostEqual(summary["test", "tur_Latn"]["recovery_fraction"], 2 / 3)
        self.assertIsNone(summary["test", "jpn_Jpan"]["recovery_fraction"])
        self.assertIsNone(summary["validation", "all"]["recovery_fraction"])

    def test_recovery_is_signed_and_may_exceed_one(self):
        for bits, expected in ((20, -1.0), (5, 2.0)):
            with self.subTest(bits=bits):
                cells = {cell: rows[:1] for cell, rows in examples().items()}
                cells[TRAINED][0]["logp"] = -bits * math.log(2)
                rows = hybrid_training_summary(cells)
                self.assertAlmostEqual(rows[0]["recovery_fraction"], expected)

    def test_checks_pairing_bytes_and_matched_inference_tokens(self):
        for cell, change, message in (
            (INFERENCE_ONLY, lambda rows: rows.pop(), "Different document set"),
            (INFERENCE_ONLY, lambda rows: rows.append(rows[0]), "Duplicate routing document"),
            (TRAINED, lambda rows: rows[0].update(bytes=11), "Different byte count"),
            (TRAINED, lambda rows: rows[0].update(tokens=6), "Hybrid inference token counts differ"),
            (TRAINED, lambda rows: rows[0].update(logp=float("inf")), "Invalid log probability"),
        ):
            with self.subTest(message=message):
                cells = examples()
                change(cells[cell])
                with self.assertRaisesRegex(ValueError, message):
                    hybrid_training_summary(cells)


if __name__ == "__main__":
    unittest.main()
