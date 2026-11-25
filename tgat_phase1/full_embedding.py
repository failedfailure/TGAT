"""Enable an FP32 master for the entire tied embedding table."""

import torch
from torch import nn
from torch.nn import functional as F


class FullEmbedding(nn.Embedding):
    def __init__(self, source, weight):
        super().__init__(source.num_embeddings, source.embedding_dim,
                         padding_idx=source.padding_idx, max_norm=source.max_norm,
                         norm_type=source.norm_type, scale_grad_by_freq=source.scale_grad_by_freq,
                         sparse=source.sparse, device="meta")
        self.compute_dtype = source.weight.dtype
        self.weight = weight

    def forward(self, input):
        # Gather before casting so the input path casts only the requested rows.
        return super().forward(input).to(self.compute_dtype)

    def save(self, path):
        from safetensors.torch import save_file

        weight = self.weight.detach().cpu().contiguous()
        if weight.dtype != torch.float32 or not torch.isfinite(weight).all():
            raise ValueError("Expected finite FP32 master embedding weights")
        save_file({"weight": weight}, str(path))


class _TiedProjection(nn.Linear):
    def __init__(self, source, weight):
        super().__init__(source.in_features, source.out_features, bias=False, device="meta")
        self.weight = weight

    def forward(self, input):
        return F.linear(input, self.weight.to(input.dtype))


def _attach(model, saved, trainable):
    embedding = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if (not isinstance(embedding, nn.Embedding) or not isinstance(head, nn.Linear)
            or isinstance(embedding, FullEmbedding) or head.bias is not None):
        raise ValueError("Full embedding adaptation requires an unmodified embedding and bias-free projection")
    if embedding.weight is not head.weight or embedding.weight.requires_grad:
        raise ValueError("Input embeddings and the output projection must share one frozen base parameter")

    weight = embedding.weight.detach()
    if saved is None:
        master = weight.to(dtype=torch.float32, copy=True)
    else:
        if saved.shape != weight.shape or saved.dtype != torch.float32 or not torch.isfinite(saved).all():
            raise ValueError("Saved embedding weights must have the expected shape and finite FP32 values")
        master = saved.to(device=weight.device, copy=True)
    parameter = nn.Parameter(master, requires_grad=trainable)
    replacement = FullEmbedding(embedding, parameter)
    model.set_input_embeddings(replacement)
    model.set_output_embeddings(_TiedProjection(head, parameter))
    if model.get_output_embeddings().weight is not model.get_input_embeddings().weight:
        raise RuntimeError("Full embedding adaptation lost input and output weight tying")
    return replacement


def attach_full_embedding(model, *, trainable=True):
    """Replace a frozen tied table after PEFT wrapping, retaining the model's compute dtype."""
    return _attach(model, saved=None, trainable=trainable)


def load_full_embedding(model, path, *, trainable=False):
    """Restore a saved FP32 master and retie the input and output after PEFT reload."""
    from safetensors.torch import load_file

    saved = load_file(str(path))
    if saved.keys() != {"weight"}:
        raise ValueError("Expected one shared weight tensor in the full embedding artifact")
    return _attach(model, saved["weight"], trainable=trainable)
