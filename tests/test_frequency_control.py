import copy
import unittest
from collections import Counter

from tgat_phase1.frequency_control import frequency_controls
from tgat_phase1.row_selection import select_rows


class FrequencyControlTests(unittest.TestCase):
    def setUp(self):
        self.languages = ("en", "ja")
        self.ordinary = range(8)
        original = (
            [1] * 40 + [2] * 19 + [3] * 17 + [6] * 80 + [99] * 200,
            [4] * 30 + [5] * 5 + [6] * 20 + [7] * 3,
        )
        self.target = (
            [0] * 55 + [1] * 73 + [2] * 19 + [3] * 17 + [6] * 30 + [99] * 200,
            [4] * 72 + [5] * 55 + [6] * 30 + [7] * 3,
        )
        self.seeds = [42, 43, 44]
        self.plan = {"size": 2, "min_support": 16, "seeds": self.seeds, "draws": {
            str(seed): select_rows(original, self.target, self.ordinary, size=2, min_support=16, seed=seed)
            for seed in self.seeds
        }}

    def test_matches_count_bins_and_dominant_languages(self):
        plan = frequency_controls(self.target, self.languages, self.ordinary, self.plan)
        self.assertEqual(plan, frequency_controls(iter(self.target), iter(self.languages),
                                                  reversed(self.ordinary), self.plan))
        eligible_strata = {(r["minimum_g3_count"], r["dominant_language"]): r for r in plan["strata"]}
        self.assertEqual(eligible_strata[32, "en"]["eligible_rows"], 2)

        language_counts = {lang: Counter(ids) for lang, ids in zip(self.languages, self.target, strict=True)}
        total_counts = sum(language_counts.values(), Counter())

        def strata(ids):
            return Counter(
                (total_counts[i].bit_length(), max(sorted(language_counts), key=lambda lang: language_counts[lang][i]))
                for i in ids
            )

        g4 = self.plan["draws"]["42"]["selected_ids"]
        self.assertEqual(g4, [0, 5])
        for seed in self.seeds:
            draw = plan["draws"][str(seed)]
            matched = draw["matched_ids"]
            self.assertEqual(matched, sorted(set(matched)))
            self.assertEqual(strata(matched), strata(g4))
            self.assertEqual(draw["high_frequency_ids"], [1, 4])
            self.assertEqual(draw["matched"]["target_counts"], [total_counts[i] for i in matched])
            self.assertEqual(draw["matched"]["overlap_with_g4"], len(set(matched).intersection(g4)))
            for language, counts in language_counts.items():
                coverage = draw["matched"]["coverage_by_language"][language]
                self.assertEqual(coverage["selected_occurrences"], sum(counts[i] for i in matched))
                self.assertEqual(coverage["training_tokens"], sum(counts.values()))

    def test_verifies_original_plan_for_each_seed(self):
        changed = copy.deepcopy(self.plan)
        changed["draws"]["43"]["sha256"]["eligible_rows"] = "unexpected"
        with self.assertRaisesRegex(ValueError, "for seed 43"):
            frequency_controls(self.target, self.languages, self.ordinary, changed)
        with self.assertRaisesRegex(ValueError, "for seed 42"):
            frequency_controls(self.target, self.languages, (*self.ordinary, 99), self.plan)
        with self.assertRaises(ValueError):
            frequency_controls(self.target, self.languages[:1], self.ordinary, self.plan)
        changed_paths = (list(reversed(self.target[0])), self.target[1])
        with self.assertRaisesRegex(ValueError, "for seed 42"):
            frequency_controls(changed_paths, self.languages, self.ordinary, self.plan)


if __name__ == "__main__":
    unittest.main()
