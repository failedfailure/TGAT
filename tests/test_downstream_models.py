from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tgat_phase1 import downstream_models


class DownstreamModelsTests(unittest.TestCase):
    def test_original_and_native_langmap_use_expected_language_and_normalized_text(self):
        router = downstream_models.DownstreamModels("unused")
        original, native, ignored = Mock(), Mock(), Mock()
        router._original = SimpleNamespace(backend_tokenizer=original)
        router._native = {"eng_Latn": ignored, "tur_Latn": native}
        for tokenizer, ids in ((original, [1]), (native, [2, 3])):
            tokenizer.encode.return_value = SimpleNamespace(ids=ids)
            tokenizer.decode.return_value = "é\n  "

        for group in ("G0", "G1"):
            self.assertEqual(router.encode("e\u0301\r\n  ", "eng_Latn", group), [1])
        self.assertEqual(router.encode("e\u0301\r\n  ", "tur_Latn", "G2"), [2, 3])
        original.encode.assert_called_with("é\n  ", add_special_tokens=False)
        native.encode.assert_called_once_with("é\n  ", add_special_tokens=False)
        native.decode.assert_called_once_with([2, 3], skip_special_tokens=False)
        ignored.encode.assert_not_called()
        with self.assertRaisesRegex(ValueError, "label"):
            router.encode("example", "und", "G0")
        with self.assertRaisesRegex(ValueError, "Unknown downstream group"):
            router.encode("example", "eng_Latn", "G7")

    def test_boundary_models_share_the_frozen_lambda_and_language_hints(self):
        router = downstream_models.DownstreamModels("unused")
        segmenter = Mock()
        segmenter.encode.return_value = (77, 28)
        stems = object()
        router._boundary_encoder = ({"jpn_Jpan": segmenter}, stems, 8.0)
        with patch.object(downstream_models, "boundary_hints", return_value={2: 0.25}) as hints:
            for group in ("G3", "G4", "G5", "G6", "frequency-matched"):
                self.assertEqual(router.encode("静かな\r\n", "jpn_Jpan", group), [77, 28])
        hints.assert_called_with("静かな\n", "jpn_Jpan", stems)
        self.assertEqual(segmenter.encode.call_count, 5)
        segmenter.encode.assert_called_with("静かな\n", {2: 2.0})

    def test_lo_ra_is_restored_before_either_embedding_sidecar(self):
        router = downstream_models.DownstreamModels("unused", device="cpu", dtype="float32")
        router._pilot = {"eos_id": 5, "packages": {}}
        artifact = Path("adapter-dir")
        base = SimpleNamespace(config=SimpleNamespace(eos_token_id=5))
        restored = Mock()
        restored.requires_grad_.return_value = restored
        restored.eval.return_value = restored
        calls = Mock()
        calls.lora.return_value = restored
        full = SimpleNamespace(load_full_embedding=calls.full)
        tied = SimpleNamespace(load_tied_rows=calls.tied)
        with (patch.object(downstream_models, "load_model", return_value=base),
              patch.object(downstream_models, "saved_model", calls.lora),
              patch.dict(sys.modules, {"tgat_phase1.full_embedding": full, "tgat_phase1.tied_rows": tied})):
            with patch.object(router, "_resolve", return_value=(artifact, None, {})):
                self.assertIs(router.load("G5"), restored)
            with patch.object(router, "_resolve", return_value=(artifact, [3, 6], {})):
                self.assertIs(router.load("G4"), restored)

        self.assertEqual([call[0] for call in calls.mock_calls],
                         ["lora", "full", "lora().requires_grad_", "lora().eval",
                          "lora", "tied", "lora().requires_grad_", "lora().eval"])
        calls.full.assert_called_once_with(restored, artifact / "embedding.safetensors")
        calls.tied.assert_called_once_with(restored, artifact / "rows.safetensors", [3, 6])


if __name__ == "__main__":
    unittest.main()
