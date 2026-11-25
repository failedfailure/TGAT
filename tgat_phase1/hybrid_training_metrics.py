"""Summarize training under the same inference routing policy."""

from collections import defaultdict
import math

from .routing_metrics import HYBRID, INPUT_CELLS, POLICY, _index


BASELINE = INPUT_CELLS[0]
INFERENCE_ONLY = HYBRID
TRAINED = "G4-hybrid-trained"
CELLS = (BASELINE, INFERENCE_ONLY, TRAINED)


def _paired_cells(cells):
    if set(cells) != set(CELLS):
        raise ValueError(f"Expected exactly the three hybrid-training cells: {CELLS}")
    index = {cell: _index(cells[cell], cell) for cell in CELLS}
    reference = index[BASELINE]
    if not reference:
        raise ValueError("Hybrid training comparison requires at least one scored document")
    for cell in CELLS[1:]:
        if index[cell].keys() != reference.keys():
            raise ValueError(f"Different document set in {cell}")
        for key, expected in reference.items():
            if index[cell][key]["bytes"] != expected["bytes"]:
                raise ValueError(f"Different byte count for {cell}: {key}")
    for key in reference:
        if index[INFERENCE_ONLY][key]["tokens"] != index[TRAINED][key]["tokens"]:
            raise ValueError(f"Hybrid inference token counts differ: {key}")
    return index


def _summarize(index, keys, split, lang):
    size = sum(index[BASELINE][key]["bytes"] for key in keys)
    denominator = size * math.log(2)
    totals = {cell: math.fsum(index[cell][key]["logp"] for key in keys) for cell in CELLS}
    result = {"split": split, "lang": lang, "documents": len(keys), "bytes": size}
    for cell in CELLS:
        prefix = cell.lower().replace("-", "_")
        tokens = sum(index[cell][key]["tokens"] for key in keys)
        result.update({f"{prefix}_bpb": -totals[cell] / denominator, f"{prefix}_tokens": tokens,
                       f"{prefix}_tokens_per_byte": tokens / size})

    baseline, inference, trained = (totals[cell] for cell in CELLS)
    gap = baseline - inference
    change = inference - trained
    result.update(inference_gap_bpb=gap / denominator, trained_gap_bpb=(baseline - trained) / denominator,
                  trained_vs_inference_bpb=change / denominator,
                  recovery_fraction=(trained - inference) / gap if gap > 0 else None)
    return result


def hybrid_training_summary(cells):
    """Report byte-weighted BPB, paired changes and recovery for each split.

    Recovery is (inference-only BPB − trained BPB) divided by the positive
    gap between inference-only BPB and G1 BPE. It can be negative or exceed
    one. It is undefined when the inference-only gap is zero or negative.
    """
    index = _paired_cells(cells)
    splits = defaultdict(lambda: defaultdict(list))
    for key in index[BASELINE]:
        split, lang, _ = key
        splits[split][lang].append(key)
        splits[split]["all"].append(key)
    return [_summarize(index, keys[lang], split, lang)
            for split, keys in sorted(splits.items()) for lang in (*POLICY, "all") if lang in keys]
