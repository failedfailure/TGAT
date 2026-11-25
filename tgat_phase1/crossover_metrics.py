"""Summarize paired accuracy in a checkpoint-by-tokenizer crossover."""

from collections import defaultdict
from collections.abc import Mapping


CELLS = (("G1", "BPE"), ("G1", "G3"), ("G4", "BPE"), ("G4", "G3"))
METRICS = ("raw", "byte")
_CONTRASTS = (
    ("tokenizer_at_g1", CELLS[0], CELLS[1]),
    ("tokenizer_at_g4", CELLS[2], CELLS[3]),
    ("checkpoint_at_bpe", CELLS[0], CELLS[2]),
    ("checkpoint_at_g3", CELLS[1], CELLS[3]),
    ("diagonal", CELLS[0], CELLS[3]),
)


def _index(records):
    indexed = {}
    for row in records.values() if isinstance(records, Mapping) else records:
        key = tuple(str(row[field]) for field in ("dataset", "lang", "phenomenon", "id"))
        if key in indexed:
            raise ValueError(f"Duplicate crossover pair: {key}")
        correctness = []
        for metric in METRICS:
            value = row[f"{metric}_correct"]
            if not (type(value) in (int, bool) and value in (0, 1) or value in ("0", "1")):
                raise ValueError(f"Invalid {metric} correctness for crossover pair: {key}")
            correctness.append(int(value))
        indexed[key] = (row["reference_overlap"], *correctness)
    return indexed


def _summarize_subset(keys, indexed, dataset, lang, subset):
    n = len(keys)
    for column, metric in enumerate(METRICS, start=1):
        values = {cell: [indexed[cell][key][column] for key in keys] for cell in CELLS}
        summary = {"dataset": dataset, "lang": lang, "subset": subset, "metric": metric, "pairs": n}
        totals = {cell: sum(correct) for cell, correct in values.items()}
        for (checkpoint, tokenizer), correct in totals.items():
            summary[f"{checkpoint.lower()}_{tokenizer.lower()}_pct"] = 100 * correct / n
        for contrast, baseline, target in _CONTRASTS:
            changes = [after - before for before, after in zip(values[baseline], values[target], strict=True)]
            wins, losses = changes.count(1), changes.count(-1)
            summary.update({f"{contrast}_pp": 100 * (wins - losses) / n,
                            f"{contrast}_wins": wins, f"{contrast}_losses": losses})
        a, b, c, d = (totals[cell] for cell in CELLS)
        summary["interaction_pp"] = 100 * ((d - c) - (b - a)) / n
        yield summary


def crossover_summary(cells):
    """Return dataset/language/subset accuracy in percent and paired effects in pp.

    Tokenizer effects are G3 minus BPE at each checkpoint. Checkpoint effects
    are G4 minus G1 at each tokenizer. The interaction is the difference of
    tokenizer effects, G4 minus G1. In a contrast, a win changes an incorrect
    pair to correct and a loss changes a correct pair to incorrect.
    """
    if set(cells) != set(CELLS):
        raise ValueError(f"Expected exactly the four crossover cells: {CELLS}")
    indexed = {cell: _index(cells[cell]) for cell in CELLS}
    anchor = indexed[CELLS[0]]
    if not anchor:
        raise ValueError("Crossover requires at least one scored pair")
    for cell in CELLS[1:]:
        if indexed[cell].keys() != anchor.keys():
            raise ValueError(f"Different scored pair set in crossover cell {cell}")
        for key, (overlap, *_) in anchor.items():
            if indexed[cell][key][0] != overlap:
                raise ValueError(f"Reference overlap differs in crossover cell {cell} for {key}")

    suites = defaultdict(list)
    for key in anchor:
        suites[key[:2]].append(key)
    result = []
    for (dataset, lang), keys in sorted(suites.items()):
        result.extend(_summarize_subset(keys, indexed, dataset, lang, "full"))
        outside = [key for key in keys if not anchor[key][0]]
        if 0 < len(outside) < len(keys):
            result.extend(_summarize_subset(outside, indexed, dataset, lang, "outside-reference"))
    return result
