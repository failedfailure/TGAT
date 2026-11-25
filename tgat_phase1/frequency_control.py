"""Derive same-budget frequency controls from a frozen G4/G6 embedding plan."""

import hashlib
import random
from collections import Counter, defaultdict
from statistics import median

from .row_selection import _row_digest, _update_digest


RNG_DOMAIN = "tgat-frequency-language-matched-v1"


def frequency_controls(training_ids, languages, ordinary_ids, source_plan):
    """Match G4's allocation across G3 count bins and dominant languages.

    The input paths and vocabulary must reproduce the original plan's
    fingerprints. Document languages must align with the paths. Each seed
    independently samples ordinary eligible rows without replacement within
    each powers-of-two frequency bin and dominant training language. G4
    remains in the pool. The secondary high-frequency arm takes the most
    used eligible rows.
    """
    ordinary = sorted(set(ordinary_ids))
    size = source_plan["size"]
    support = source_plan["min_support"]
    counts = Counter()
    by_language = defaultdict(Counter)
    paths = hashlib.sha256()
    language_order = hashlib.sha256()
    documents = 0
    for ids, language in zip(training_ids, languages, strict=True):
        counts.update(ids)
        by_language[language].update(ids)
        _update_digest(paths, ids)
        language_order.update(language.encode("utf-8") + b"\x00")
        documents += 1

    eligible = [row for row in ordinary if counts[row] >= support]
    eligible_set = set(eligible)
    seeds = source_plan["seeds"]
    selected = source_plan["draws"][str(seeds[0])]["selected_ids"]
    selected_set = set(selected)
    if not size or selected != sorted(selected_set) or len(selected) != size or selected_set - eligible_set:
        raise ValueError("G4 rows must be sorted, unique, and eligible at the original row budget")

    fingerprints = {
        "ordinary_rows": _row_digest(ordinary),
        "eligible_rows": _row_digest(eligible),
        "target_paths": paths.hexdigest(),
        "selected_ids": _row_digest(selected),
    }
    expected = {
        "size": size, "min_support": support, "support_basis": "target",
        "documents": documents, "ordinary_rows": len(ordinary), "eligible_rows": len(eligible),
        "target_total": sum(counts[row] for row in ordinary), "selected_ids": selected,
    }
    for seed in seeds:
        source = source_plan["draws"][str(seed)]
        if (source["selection_seed"] != seed or any(source.get(k) != v for k, v in expected.items())
                or any(source["sha256"].get(k) != v for k, v in fingerprints.items())):
            raise ValueError(f"Frozen G4/G6 plan differs from the G3 paths, vocabulary, or selected rows for seed {seed}")

    language_names = sorted(by_language)
    language_totals = {language: sum(by_language[language].values()) for language in language_names}

    def stratum(row):
        # max returns the first item on ties, so prefer the smallest language code.
        dominant = max(language_names, key=lambda language: by_language[language][row])
        return counts[row].bit_length() - 1, dominant

    pools = defaultdict(list)
    for row in eligible:
        pools[stratum(row)].append(row)
    allocation = Counter(map(stratum, selected))
    high_frequency = sorted(sorted(eligible, key=lambda row: (-counts[row], row))[:size])

    def describe(rows):
        occurrences = [counts[row] for row in rows]
        language_occurrences = {
            language: sum(by_language[language][row] for row in rows) for language in language_names
        }
        return {
            "sha256": _row_digest(rows), "target_counts": occurrences,
            "target_occurrences": sum(occurrences), "median_count": median(occurrences),
            "overlap_with_g4": len(selected_set.intersection(rows)),
            "coverage_by_language": {
                language: {
                    "selected_occurrences": language_occurrences[language],
                    "training_tokens": language_totals[language],
                    "fraction": language_occurrences[language] / language_totals[language],
                }
                for language in language_names
            },
        }

    draws = {}
    for seed in seeds:
        rng = random.Random(f"{RNG_DOMAIN}:{seed}")
        matched = sorted(row for key, n in sorted(allocation.items()) for row in rng.sample(pools[key], n))
        draws[str(seed)] = {
            "matched_ids": matched, "high_frequency_ids": high_frequency,
            "matched": describe(matched), "high_frequency": describe(high_frequency),
        }
    return {
        "method": RNG_DOMAIN,
        "eligible_rows": len(eligible),
        "source_sha256": fingerprints,
        "language_order_sha256": language_order.hexdigest(),
        "g4": describe(selected),
        "strata": [
            {"minimum_g3_count": 2 ** low, "maximum_g3_count": 2 ** (low + 1) - 1,
             "dominant_language": language, "eligible_rows": len(pool),
             "rows_per_matched_draw": allocation[low, language]}
            for (low, language), pool in sorted(pools.items())
        ],
        "draws": draws,
    }
