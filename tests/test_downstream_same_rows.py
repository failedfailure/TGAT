from pathlib import Path
import unittest
from unittest.mock import patch

from reproduction.downstream_same_rows import GROUP, SameRowsModels
from tgat_phase1.downstream_models import DownstreamModels


class SameRowsModelsTests(unittest.TestCase):
    def test_routes_the_matched_checkpoint_through_native_langmap(self):
        router = SameRowsModels(Path("run"), Path("matched"))
        with patch.object(DownstreamModels, "encode", return_value=[13]) as encode:
            self.assertEqual(router.encode("text", "tur_Latn", GROUP), [13])
            encode.assert_called_once_with("text", "tur_Latn", "G2")

    def test_rejects_a_different_row_set_before_loading_weights(self):
        router = SameRowsModels(Path("run"), Path("matched"))
        ids = list(range(512))
        g4 = {"g3_selection_sha256": "selection", "training_bpb_sha256": "g4-bpb",
              "artifacts_sha256": {"rows.safetensors": "g4-rows"}}
        pilot, g2, g3 = {"seed": 42}, Path("g2"), Path("g3")
        router._plan = {"training_budget": {"train_tokens": 3}}
        manifest = {
            "pilot": pilot,
            "unit": "one pass over the same raw training documents",
            "selection_sha256": "selection",
            "row_plan_sha256": "sha",
            "reference_scores_sha256": {"g2_g3": "sha", "g4_g6": "g4-bpb"},
            "g4_weights_sha256": g4["artifacts_sha256"],
            "row_ids": list(range(1, 513)),
            "g3_budget": router._plan["training_budget"],
            "smoke": False,
        }
        with (patch.object(DownstreamModels, "_resolve",
                           side_effect=[(Path("g4"), ids, g4), (g2, None, {})]),
              patch.object(router, "_pilot_seed", return_value=(pilot, g2)),
              patch.object(router, "_g3_run", return_value=(g3, {}, {})),
              patch("reproduction.downstream_same_rows.read_json", return_value=manifest),
              patch("reproduction.downstream_same_rows.sha256_file", return_value="sha"),
              patch("reproduction.downstream_same_rows.verified_group") as verify):
            with self.assertRaisesRegex(ValueError, "manifest differs at row_ids"):
                router.provenance(GROUP)
            verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
