import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required")
class TiedRowsTests(unittest.TestCase):
    @staticmethod
    def model(dtype, device="cpu"):
        import torch
        from torch import nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(7, 4, dtype=dtype, device=device)
                self.head = nn.Linear(4, 7, bias=False, dtype=dtype, device=device)
                self.head.weight = self.embedding.weight
                self.embedding.weight.requires_grad_(False)

            def get_input_embeddings(self):
                return self.embedding

            def get_output_embeddings(self):
                return self.head

            def forward(self, ids):
                return self.head(self.embedding(ids))

        model = Model()
        with torch.no_grad():
            model.embedding.weight.copy_(torch.arange(28).reshape(7, 4) / 32)
        return model

    def test_forward_and_both_gradient_contributions_match_a_tied_table(self):
        import torch
        from torch.nn import functional as F

        from tgat_phase1.tied_rows import attach_tied_rows

        ids = torch.tensor([[0, 2, 3, 5, 6]])
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = self.model(dtype)
                original = model.embedding.weight.detach().clone()
                before = model(ids)
                rows = attach_tied_rows(model, [2, 5])
                torch.testing.assert_close(model(ids), before, atol=0, rtol=0)
                with torch.no_grad():
                    rows.delta.copy_(torch.tensor([[0.25, -0.125, 0.5, 0.125], [-0.5, 0.5, 0.25, 0]]))

                expected_delta = rows.delta.detach().clone().requires_grad_(True)
                effective = (original.index_select(0, rows.ids).float() + expected_delta).to(dtype)
                expected = original.index_copy(0, rows.ids, effective)
                expected_logits = F.linear(F.embedding(ids, expected), expected)
                actual = model(ids)
                tolerances = {"atol": 4e-3, "rtol": 2e-2} if dtype == torch.bfloat16 else {}
                torch.testing.assert_close(actual, expected_logits, **tolerances)
                labels = torch.tensor([1, 5, 6, 2, 4])
                F.cross_entropy(actual[0].float(), labels).backward()
                F.cross_entropy(expected_logits[0].float(), labels).backward()
                torch.testing.assert_close(rows.delta.grad, expected_delta.grad, atol=4e-3, rtol=3e-2)
                self.assertEqual([p.numel() for p in model.parameters() if p.requires_grad], [8])
                self.assertIs(model.embedding.weight, model.head.weight)
                self.assertIsNone(model.embedding.weight.grad)
                delta_before = rows.delta.detach().clone()
                torch.optim.AdamW([rows.delta], lr=0.05).step()
                self.assertFalse(torch.equal(delta_before, rows.delta))
                torch.testing.assert_close(model.embedding.weight, original, atol=0, rtol=0)

    def assert_zero_delta_preserves_logits(self, device, dtype):
        import torch

        from tgat_phase1.tied_rows import attach_tied_rows

        model = self.model(dtype, device)
        # Only the output path can supply a gradient for these input IDs.
        ids = torch.tensor([[0, 1, 3, 6]], device=device)
        before = model(ids)
        rows = attach_tied_rows(model, [2, 5])
        after = model(ids)
        torch.testing.assert_close(after, before, atol=0, rtol=0)
        after[..., 5].float().sum().backward()
        self.assertTrue(torch.count_nonzero(rows.delta.grad[1]).item() > 0)
        self.assertEqual(torch.count_nonzero(rows.delta.grad[0]).item(), 0)

    def test_zero_delta_preserves_logits_and_has_output_gradient_on_cpu(self):
        import torch

        self.assert_zero_delta_preserves_logits("cpu", torch.float32)

    def test_zero_delta_preserves_logits_and_has_output_gradient_on_cuda(self):
        import torch

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA BF16 is required")
        self.assert_zero_delta_preserves_logits("cuda", torch.bfloat16)

    @unittest.skipUnless(importlib.util.find_spec("safetensors"), "safetensors is required")
    def test_reload_reproduces_logits_and_rejects_unexpected_ids(self):
        import torch

        from tgat_phase1.tied_rows import attach_tied_rows, load_tied_rows

        model = self.model(torch.float32)
        rows = attach_tied_rows(model, [2, 5])
        with torch.no_grad():
            rows.delta.fill_(0.25)
        ids = torch.tensor([[0, 2, 5, 6]])
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "tied_rows.safetensors"
            rows.save(path)
            restored = self.model(torch.float32)
            with self.assertRaisesRegex(ValueError, "differ from"):
                load_tied_rows(restored, path, [3, 5])
            reloaded = load_tied_rows(restored, path, [2, 5])
            self.assertFalse(reloaded.delta.requires_grad)
            torch.testing.assert_close(restored(ids), model(ids), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
