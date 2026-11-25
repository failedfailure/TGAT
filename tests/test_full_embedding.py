import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required")
class FullEmbeddingTests(unittest.TestCase):
    @staticmethod
    def model(device="cpu"):
        import torch
        from torch import nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(7, 4, dtype=torch.bfloat16, device=device)
                self.head = nn.Linear(4, 7, bias=False, dtype=torch.bfloat16, device=device)
                self.head.weight = self.embedding.weight
                self.embedding.weight.requires_grad_(False)

            def get_input_embeddings(self):
                return self.embedding

            def get_output_embeddings(self):
                return self.head

            def set_input_embeddings(self, embedding):
                self.embedding = embedding

            def set_output_embeddings(self, head):
                self.head = head

            def forward(self, ids):
                return self.head(self.embedding(ids))

        model = Model()
        with torch.no_grad():
            model.embedding.weight.copy_(torch.arange(28).reshape(7, 4) / 32)
        return model

    def assert_trainable_table(self, device):
        import torch

        from tgat_phase1.full_embedding import attach_full_embedding

        model = self.model(device)
        ids = torch.tensor([[0, 2, 6]], device=device)
        baseline = model(ids)
        rng = torch.random.get_rng_state()
        embedding = attach_full_embedding(model)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        self.assertIs(embedding.weight, model.head.weight)
        self.assertEqual(embedding.weight.dtype, torch.float32)
        self.assertEqual(list(model.parameters()), [embedding.weight])
        self.assertEqual(model.embedding(ids).dtype, torch.bfloat16)
        torch.testing.assert_close(model(ids), baseline, atol=0, rtol=0)

        original = embedding.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        model(ids)[..., 5].float().sum().backward()
        # Input-only IDs and an output-only ID all update the same master tensor.
        self.assertGreater(torch.count_nonzero(embedding.weight.grad[2]).item(), 0)
        self.assertGreater(torch.count_nonzero(embedding.weight.grad[5]).item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(embedding.weight, original))

    def test_fp32_master_and_bf16_compute(self):
        self.assert_trainable_table("cpu")

    def test_fp32_master_and_bf16_compute_cuda(self):
        import torch

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA BF16 is required")
        self.assert_trainable_table("cuda")

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("peft", "transformers")),
                         "PEFT and Transformers are required")
    def test_qwen_lora_keeps_embedding_out_of_adapter_state(self):
        import torch
        from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
        from transformers import Qwen3Config, Qwen3ForCausalLM

        from tgat_phase1.full_embedding import attach_full_embedding

        config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                             tie_word_embeddings=True, use_cache=False)
        base = Qwen3ForCausalLM(config).to(dtype=torch.bfloat16)
        model = get_peft_model(base, LoraConfig(r=2, target_modules=["q_proj"], task_type=TaskType.CAUSAL_LM))
        ids = torch.tensor([[0, 2, 6, 7]])
        before = model(input_ids=ids).logits
        attached = attach_full_embedding(model)
        torch.testing.assert_close(model(input_ids=ids).logits, before, atol=0, rtol=0)
        model(input_ids=ids, labels=ids).loss.backward()
        self.assertIsNotNone(attached.weight.grad)
        state = get_peft_model_state_dict(model, save_embedding_layers=False)
        self.assertTrue(state)
        self.assertTrue(all("lora_" in key for key in state))

    @unittest.skipUnless(importlib.util.find_spec("safetensors"), "safetensors is required")
    def test_save_reload_and_validate_master(self):
        import torch
        from safetensors.torch import save_file

        from tgat_phase1.full_embedding import attach_full_embedding, load_full_embedding

        model = self.model()
        trained = attach_full_embedding(model)
        with torch.no_grad():
            trained.weight[2].add_(0.25)
        ids = torch.tensor([[0, 2, 6]])
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "embedding.safetensors"
            trained.save(path)
            restored = self.model()
            reloaded = load_full_embedding(restored, path)
            self.assertFalse(reloaded.weight.requires_grad)
            self.assertIs(reloaded.weight, restored.head.weight)
            torch.testing.assert_close(restored(ids), model(ids), atol=0, rtol=0)

            for invalid in (torch.ones(7, 3), torch.ones(7, 4, dtype=torch.bfloat16),
                            torch.full((7, 4), torch.nan)):
                with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                    save_file({"weight": invalid}, str(path))
                    with self.assertRaisesRegex(ValueError, "shape and finite FP32"):
                        load_full_embedding(self.model(), path)


if __name__ == "__main__":
    unittest.main()
