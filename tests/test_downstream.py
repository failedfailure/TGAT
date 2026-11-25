import importlib.util
from types import SimpleNamespace
import unittest

from tgat_phase1.downstream import aggregate, score_margins, score_sequences


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required")
class CausalScoreTests(unittest.TestCase):
    def test_targets_use_previous_tokens_and_ignore_right_padding(self):
        import torch

        class ToyModel(torch.nn.Module):
            def forward(self, input_ids, attention_mask, use_cache):
                if use_cache:
                    raise AssertionError("Benchmark should disable the KV cache")
                favorite = (input_ids + 1) % 6
                logits = torch.nn.functional.one_hot(favorite, 6).float() * 3
                logits = logits * attention_mask.unsqueeze(-1)
                return SimpleNamespace(logits=logits)

        model = ToyModel()
        eos = 1
        paths = [[2, 3, 4], [2], [5, 1]]
        expected = []
        for ids in paths:
            prior = torch.tensor([[eos, *ids[:-1]]])
            logits = model(prior, torch.ones_like(prior), use_cache=False).logits
            positions = torch.arange(len(ids))
            expected.append(torch.log_softmax(logits, dim=-1)[0, positions, ids].sum(dtype=torch.float64))
        for batch_size in (1, 8):
            actual = score_sequences(model, paths, eos, torch.device("cpu"), batch_size)
            torch.testing.assert_close(torch.tensor(actual, dtype=torch.float64), torch.stack(expected))

    def test_context_budget_rejects_truncation(self):
        with self.assertRaisesRegex(ValueError, "complete-sentence scoring"):
            score_sequences(None, [[1] * 513], 0, "cpu", 1)


class PairMetricsTests(unittest.TestCase):
    def test_aggregates_ties_and_byte_normalization(self):
        row = {"good_logp": -4, "bad_logp": -4, "good_bytes": 8, "bad_bytes": 4,
               "dataset": "multiblimp", "lang": "eng_Latn", "phenomenon": "agreement", "reference_overlap": ""}
        for metric, margin in score_margins(row).items():
            row[f"{metric}_margin"] = margin
            row[f"{metric}_correct"] = int(margin > 0)
        report = next(x for x in aggregate([row]) if x["phenomenon"] == "all")
        self.assertEqual(report["pairs"], 1)
        self.assertEqual(report["raw_ties"], 1)
        self.assertEqual(report["raw_accuracy"], 0)
        self.assertEqual(report["byte_accuracy"], 1)

        overlapping = {**row, "reference_overlap": "validation", "raw_margin": 1.0, "raw_correct": 1}
        subset = {r["subset"]: r for r in aggregate([row, overlapping]) if r["phenomenon"] == "all"}
        self.assertEqual(subset["full"]["raw_accuracy"], 0.5)
        self.assertEqual(subset["outside-reference"]["raw_accuracy"], 0)


if __name__ == "__main__":
    unittest.main()
