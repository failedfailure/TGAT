import math
import unittest

from tgat_phase1.routing_metrics import HYBRID, INPUT_CELLS, hybrid_rows, routing_summary


def scored_cells():
    sizes = (10, 30, 60, 12)
    languages = ("eng_Latn", "jpn_Jpan", "tur_Latn", "jpn_Jpan")
    bits = ((12, 24, 90, 12), (10, 30, 72, 15), (18, 33, 66, 18))
    bpe_tokens = (5, 10, 30, 6)
    token_counts = (bpe_tokens, bpe_tokens, (7, 11, 25, 7))
    return {
        cell: [{"id": f"doc-{i}", "split": "validation" if i == 3 else "test", "lang": lang,
                "bytes": size, "tokens": count, "logp": -cost * math.log(2)}
               for i, (size, lang, cost, count) in enumerate(zip(sizes, languages, sums, tokens, strict=True))]
        for cell, sums, tokens in zip(INPUT_CELLS, bits, token_counts, strict=True)
    }


class RoutingMetricsTests(unittest.TestCase):
    def test_hybrid_uses_existing_rows_and_uses_byte_weighted_totals_and_deltas(self):
        cells = scored_cells()
        cells["G4-G3"].reverse()
        hybrid = hybrid_rows(cells)
        self.assertEqual([row["source_cell"] for row in hybrid], ["G4-BPE", "G4-BPE", "G4-G3", "G4-BPE"])
        summary = {(row["split"], row["lang"], row["cell"]): row for row in routing_summary(cells)}
        total = summary["test", "all", HYBRID]
        self.assertEqual((total["documents"], total["bytes"], total["tokens"]), (3, 100, 40))
        self.assertAlmostEqual(total["bpb"], 1.06)
        self.assertAlmostEqual(total["tokens_per_byte"], 0.40)
        for reference, bpb_delta, token_delta in (("g1_bpe", -0.20, -0.05), ("g4_g3", -0.11, -0.03),
                                                  ("g4_bpe", -0.06, -0.05)):
            with self.subTest(reference=reference):
                self.assertAlmostEqual(total[f"bpb_vs_{reference}"], bpb_delta)
                self.assertAlmostEqual(total[f"tokens_per_byte_vs_{reference}"], token_delta)
        self.assertEqual(summary["test", "tur_Latn", HYBRID]["bpb_vs_g4_g3"], 0)
        self.assertEqual(summary["test", "eng_Latn", HYBRID]["bpb_vs_g4_bpe"], 0)
        self.assertEqual(summary["validation", "all", HYBRID]["documents"], 1)

    def test_input_numbers_can_be_read_from_csv(self):
        cells = scored_cells()
        expected = routing_summary(cells)
        for rows in cells.values():
            for row in rows:
                for field in ("bytes", "tokens", "logp"):
                    row[field] = str(row[field])
        self.assertEqual(routing_summary(cells), expected)
        for cell, rows in cells.items():
            for i, row in enumerate(rows):
                row["id"] = i if cell == "G1-BPE" else str(i)
        self.assertEqual(routing_summary(cells), expected)

    def test_rejects_unpaired_invalid_or_inconsistent_rows(self):
        for change, message in (
            (lambda rows: rows.pop(), "Different routing document set"),
            (lambda rows: rows.append(rows[0]), "Duplicate routing document"),
            (lambda rows: rows[0].update(bytes=99), "Different byte count"),
            (lambda rows: rows[0].update(tokens=3), "BPE token counts differ"),
            (lambda rows: rows[0].update(logp="NaN"), "Invalid log probability"),
            (lambda rows: rows[0].update(lang="und"), "Invalid routing document key"),
        ):
            with self.subTest(message=message):
                cells = scored_cells()
                change(cells["G4-BPE"])
                with self.assertRaisesRegex(ValueError, message):
                    routing_summary(cells)


if __name__ == "__main__":
    unittest.main()
