"""Apply a small trainable row update to Qwen's tied input and output weights."""

import torch
from torch import nn
from torch.nn import functional as F


class TiedRows(nn.Module):
    def __init__(self, weight, ids, delta, trainable):
        super().__init__()
        self.register_buffer("ids", ids)
        self.register_buffer("base_rows", weight.detach().index_select(0, ids), persistent=False)
        initial = torch.zeros_like(self.base_rows, dtype=torch.float32)
        if delta is not None:
            if delta.shape != initial.shape or delta.dtype != initial.dtype or not torch.isfinite(delta).all():
                raise ValueError("Invalid tied-row delta shape, precision, or values")
            initial.copy_(delta)
        self.delta = nn.Parameter(initial, requires_grad=trainable)

    def effective_rows(self):
        return (self.base_rows.float() + self.delta).to(self.base_rows.dtype)

    def correct_embeddings(self, module, args, kwargs, output):
        tokens = args[0] if args else kwargs["input"]
        positions = torch.searchsorted(self.ids, tokens.contiguous()).clamp_max(len(self.ids) - 1)
        selected = tokens == self.ids[positions]
        replacement = F.embedding(positions, self.effective_rows())
        return torch.where(selected.unsqueeze(-1), replacement, output)

    def correct_logits(self, module, args, kwargs, output):
        hidden = args[0] if args else kwargs["input"]
        # Add the change from the same rounded rows used by the input lookup.
        # The correction is exactly zero at initialization independently of the
        # GEMM shape, and only selected logits need extra projection work.
        change = self.effective_rows().float() - self.base_rows.float()
        correction = F.linear(hidden.float(), change).to(output.dtype)
        return output.index_add_(-1, self.ids, correction)

    def save(self, path):
        from safetensors.torch import save_file

        delta = self.delta.detach().cpu()
        if not torch.isfinite(delta).all():
            raise ValueError("Cannot save non-finite tied-row updates")
        save_file({"ids": self.ids.cpu(), "delta": delta}, str(path))


def attach_tied_rows(model, token_ids, *, delta=None, trainable=True):
    """Attach after PEFT wraps the model and freezes the shared base table."""
    if hasattr(model, "tied_rows"):
        raise ValueError("Tied rows are already attached")
    embedding = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if not isinstance(embedding, nn.Embedding) or not isinstance(head, nn.Linear) or head.bias is not None:
        raise ValueError("Tied rows require an embedding and a bias-free output projection")
    if embedding.weight is not head.weight or embedding.weight.requires_grad:
        raise ValueError("Input embeddings and the output projection must share one frozen parameter")

    ids = list(token_ids)
    if (not ids or any(type(index) is not int or not 0 <= index < embedding.num_embeddings for index in ids)
            or ids != sorted(set(ids))):
        raise ValueError("Tied row IDs must be nonempty, sorted, unique and within the vocabulary")
    selected = torch.tensor(ids, dtype=torch.long, device=embedding.weight.device)
    rows = TiedRows(embedding.weight, selected, delta, trainable)
    model.add_module("tied_rows", rows)
    embedding.register_forward_hook(rows.correct_embeddings, with_kwargs=True)
    head.register_forward_hook(rows.correct_logits, with_kwargs=True)
    return rows


def load_tied_rows(model, path, token_ids, *, trainable=False):
    """Reload and verify the selected rows against the experiment manifest."""
    from safetensors.torch import load_file

    saved = load_file(str(path))
    ids = list(token_ids)
    if (saved.keys() != {"ids", "delta"} or saved["ids"].dtype != torch.int64
            or saved["ids"].ndim != 1 or saved["ids"].tolist() != ids):
        raise ValueError("Saved tied-row IDs differ from the experiment manifest")
    return attach_tied_rows(model, ids, delta=saved["delta"], trainable=trainable)
