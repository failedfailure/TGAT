"""Rank embedding rows by the token frequency changes on matched training text."""

import hashlib
import math
import random
import struct
from collections import Counter


PSEUDOCOUNT = 0.5
DEFAULT_SIZE = 512
DEFAULT_MIN_SUPPORT = 32


def _update_digest(digest, ids):
    digest.update(struct.pack(f"<{len(ids) + 1}I", len(ids), *ids))


def _row_digest(ids):
    digest = hashlib.sha256()
    _update_digest(digest, ids)
    return digest.hexdigest()


def select_rows(original_paths, target_paths, ordinary_ids, *, size=DEFAULT_SIZE,
                min_support=DEFAULT_MIN_SUPPORT, seed=42):
    """Return drift ranks and a uniform control from the same eligible row pool.

    Each input iterable contains one sequence of token IDs per training document,
    in the same order. Frequencies are normalized over the supplied ordinary
    vocabulary after excluding special IDs and adding one-half to every row.
    Only rows that occur at least min_support times under the target tokenizer
    are eligible. The random draw can share rows with the drift-selected set.
    """
    if size < 1 or min_support < 1:
        raise ValueError("Row budget and minimum target support must be positive")

    ordinary_ids = set(ordinary_ids)
    original_counts = Counter()
    target_counts = Counter()
    original_digest = hashlib.sha256()
    target_digest = hashlib.sha256()
    documents = 0
    for original, target in zip(original_paths, target_paths, strict=True):
        original_counts.update(original)
        target_counts.update(target)
        _update_digest(original_digest, original)
        _update_digest(target_digest, target)
        documents += 1

    original_total = sum(original_counts[row] for row in ordinary_ids)
    target_total = sum(target_counts[row] for row in ordinary_ids)
    if not original_total or not target_total:
        raise ValueError("Both segmentations require ordinary token occurrences")
    original_denominator = original_total + PSEUDOCOUNT * len(ordinary_ids)
    target_denominator = target_total + PSEUDOCOUNT * len(ordinary_ids)
    eligible = sorted(row for row in ordinary_ids if target_counts[row] >= min_support)
    if len(eligible) < size:
        raise ValueError(f"Row budget {size} exceeds {len(eligible)} eligible rows at "
                         f"minimum target support {min_support}")

    scores = {
        row: abs(math.log(((target_counts[row] + PSEUDOCOUNT) / target_denominator)
                          / ((original_counts[row] + PSEUDOCOUNT) / original_denominator)))
        for row in eligible
    }
    ranked = sorted(eligible, key=lambda row: (-scores[row], row))
    selected_ids = sorted(ranked[:size])
    random_ids = sorted(random.Random(seed).sample(eligible, size))
    chosen = set(selected_ids) | set(random_ids)

    return {
        "size": size,
        "min_support": min_support,
        "support_basis": "target",
        "pseudocount": PSEUDOCOUNT,
        "selection_seed": seed,
        "documents": documents,
        "ordinary_rows": len(ordinary_ids),
        "eligible_rows": len(eligible),
        "original_total": original_total,
        "target_total": target_total,
        "selected_ids": selected_ids,
        "random_ids": random_ids,
        "overlap": 2 * size - len(chosen),
        "sha256": {
            "original_paths": original_digest.hexdigest(),
            "target_paths": target_digest.hexdigest(),
            "ordinary_rows": _row_digest(sorted(ordinary_ids)),
            "eligible_rows": _row_digest(eligible),
            "selected_ids": _row_digest(selected_ids),
            "random_ids": _row_digest(random_ids),
        },
        "rows": [
            {"id": row, "rank": rank, "original_count": original_counts[row],
             "target_count": target_counts[row], "absolute_log_ratio": scores[row]}
            for rank, row in enumerate(ranked, start=1) if row in chosen
        ],
    }
