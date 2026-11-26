import unittest

from reproduction.score_fineweb_single_window import (
    BASE_ROUTES, LANGUAGES, ROUTES, SPLITS, paired_contrasts, single_window_indices,
)


class SingleWindowFineWebTest(unittest.TestCase):
    def test_requires_room_for_the_prefix_in_every_tokenizer(self):
        paths = {
            "G1": [[7] * 511, [7] * 512, [7], [7], [7]],
            "G2": [[8], [8], [8] * 512, [8], [8]],
            "G3": [[9], [9], [9], [9] * 512, []],
        }
        self.assertEqual(single_window_indices(paths, 512), [0])

    def test_rejects_differently_sized_document_lists(self):
        with self.assertRaises(ValueError):
            single_window_indices({"G1": [[7]], "G2": [[8], [8]]}, 512)

    def test_reports_seed_paired_deltas_and_unreplicated_base_delta(self):
        rows = []
        for split in SPLITS:
            for lang in LANGUAGES:
                for group in BASE_ROUTES:
                    rows.append(dict(seed=None, split=split, lang=lang, group=group, bytes=99,
                                     bpb=2.0 if group == "base-G3" else 1.0))
                for seed, delta in ((42, 0.25), (43, 0.5), (44, 0.75)):
                    for group in ROUTES:
                        rows.append(dict(seed=seed, split=split, lang=lang, group=group, bytes=99,
                                         bpb=seed + delta if group == "G3" else float(seed)))

        paired, summary = paired_contrasts(rows, [42, 43, 44])
        test_all = {row["contrast"]: row for row in summary if (row["split"], row["lang"]) == ("test", "all")}
        lora = test_all["G3-minus-G2"]
        self.assertEqual((lora["bpb_difference_mean"], lora["bpb_difference_sample_sd"], lora["observations"]),
                         (0.5, 0.25, 3))
        base = test_all["base-G3-minus-base-G2"]
        self.assertEqual((base["bpb_difference_mean"], base["bpb_difference_sample_sd"], base["observations"]),
                         (1.0, "", 1))
        self.assertEqual([row["bpb_difference"] for row in paired
                          if (row["split"], row["lang"], row["contrast"]) == ("test", "all", "G3-minus-G2")],
                         [0.25, 0.5, 0.75])


if __name__ == "__main__":
    unittest.main()
