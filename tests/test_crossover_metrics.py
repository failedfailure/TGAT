import unittest

from tgat_phase1.crossover_metrics import CELLS, crossover_summary


def examples():
    accuracies = ((0, 1, 0, 1), (1, 1, 0, 0), (1, 0, 0, 1), (1, 1, 1, 0))
    return {
        cell: [{"dataset": "suite", "lang": "eng_Latn", "phenomenon": "agreement", "id": index,
                "reference_overlap": "validation" if index == 0 else "",
                "raw_correct": correct, "byte_correct": str(1 - correct)}
               for index, correct in enumerate(values)]
        for cell, values in zip(CELLS, accuracies, strict=True)
    }


class CrossoverMetricsTests(unittest.TestCase):
    def test_paired_contrasts_interaction_and_reference_sensitivity(self):
        cells = examples()
        # Reordering cells must not pair different items.
        cells[CELLS[1]] = list(reversed(cells[CELLS[1]]))
        cells[CELLS[3]] = {row["id"]: row for row in cells[CELLS[3]]}
        summary = {(row["subset"], row["metric"]): row for row in crossover_summary(cells)}
        raw = summary["full", "raw"]
        self.assertEqual(raw["pairs"], 4)
        self.assertEqual([raw[key] for key in ("g1_bpe_pct", "g1_g3_pct", "g4_bpe_pct", "g4_g3_pct")],
                         [50, 50, 50, 75])
        for contrast, (effect, wins, losses) in {
            "tokenizer_at_g1": (0, 1, 1), "tokenizer_at_g4": (25, 2, 1),
            "checkpoint_at_bpe": (0, 1, 1), "checkpoint_at_g3": (25, 1, 0),
            "diagonal": (25, 2, 1),
        }.items():
            with self.subTest(contrast=contrast):
                self.assertEqual([raw[f"{contrast}_{key}"] for key in ("pp", "wins", "losses")],
                                 [effect, wins, losses])
        self.assertEqual(raw["interaction_pp"], 25)
        self.assertEqual(summary["full", "byte"]["interaction_pp"], -25)
        self.assertEqual(summary["outside-reference", "raw"]["pairs"], 3)
        self.assertAlmostEqual(summary["outside-reference", "raw"]["interaction_pp"], 200 / 3)

    def test_suites_remain_separate_and_identical_subsets_are_omitted(self):
        cells = examples()
        for rows in cells.values():
            rows.append({**rows[0], "dataset": "other", "lang": "jpn_Jpan", "reference_overlap": ""})
        summaries = crossover_summary(cells)
        self.assertEqual([(row["lang"], row["pairs"], row["subset"]) for row in summaries if row["metric"] == "raw"],
                         [("jpn_Jpan", 1, "full"), ("eng_Latn", 4, "full"),
                          ("eng_Latn", 3, "outside-reference")])

    def test_rejects_mismatched_duplicate_and_invalid_scores(self):
        for change, message in (
            (lambda rows: rows.pop(), "Different scored pair set"),
            (lambda rows: rows.append(rows[0]), "Duplicate crossover pair"),
            (lambda rows: rows[0].update(reference_overlap="test"), "Reference overlap differs"),
            (lambda rows: rows[0].update(raw_correct=0.5), "Invalid raw correctness"),
        ):
            with self.subTest(message=message):
                cells = examples()
                change(cells[CELLS[1]])
                with self.assertRaisesRegex(ValueError, message):
                    crossover_summary(cells)


if __name__ == "__main__":
    unittest.main()
