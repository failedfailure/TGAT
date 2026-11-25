import unittest

from tgat_phase1.row_selection import select_rows


class RowSelectionTests(unittest.TestCase):
    def test_target_support_applies_to_both_groups(self):
        original = [[0] * 100 + [1] * 70 + [2, 3]]
        target = [[1] * 10 + [2] * 20 + [3] * 20 + [99] * 200]
        result = select_rows(original, target, range(4), size=2, min_support=20)
        self.assertEqual(result["eligible_rows"], 2)
        self.assertEqual(result["selected_ids"], [2, 3])
        self.assertEqual(result["random_ids"], [2, 3])
        self.assertEqual(result["target_total"], 50)
        with self.assertRaisesRegex(ValueError, "exceeds 2 eligible rows"):
            select_rows(original, target, range(4), size=3, min_support=20)


if __name__ == "__main__":
    unittest.main()
