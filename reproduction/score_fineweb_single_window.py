"""Rescore the common single-window subset of the frozen FineWeb splits."""

import argparse
from itertools import product
from pathlib import Path
from statistics import mean, stdev

from tgat_phase1 import data, train
from tgat_phase1.embedding_ablation import SEEDS, verified_runtime
from tgat_phase1.g3 import GROUPS as BOUNDARY_GROUPS, SPLITS, path_digest, publish
from tgat_phase1.g3_lora import language_totals

from .downstream_same_rows import SameRowsModels
from .g2_same_rows import GROUP as G2_ROWS


LANGUAGES = (*data.LANGUAGES, "all")
ROUTES = {"G1": "G1", "G2": "G2", "G3": "G3", "G4": "G3", "G6": "G3", G2_ROWS: "G2"}
BASE_ROUTES = {"base-G2": "G2", "base-G3": "G3"}
CONTRASTS = {
    "base-G3-minus-base-G2": ("base-G3", "base-G2"),
    "G2-minus-G1": ("G2", "G1"),
    "G2-rows-minus-G1": (G2_ROWS, "G1"),
    "G4-minus-G1": ("G4", "G1"),
    "G3-minus-G2": ("G3", "G2"),
    "G2-rows-minus-G2": (G2_ROWS, "G2"),
    "G4-minus-G3": ("G4", "G3"),
    "G4-minus-G2-rows": ("G4", G2_ROWS),
    "G4-minus-G6": ("G4", "G6"),
}


def single_window_indices(paths, context):
    """Return indices fitting one EOS-prefixed causal window on every path."""
    if context < 2 or not paths:
        raise ValueError("A causal window needs a prefix and at least one content token")
    return [
        index for index, options in enumerate(zip(*paths.values(), strict=True))
        if all(0 < len(ids) < context for ids in options)
    ]


def prepare_splits(run, router, context):
    frozen = run / "g3-eval"
    selection = data.read_json(frozen / "boundary-pilot/run.json")
    bpe_reference = data.read_json(frozen / "review-bpe/run.json")
    prepared, samples, audit, digests = {}, [], [], {}
    for split in SPLITS:
        filename = f"{split}.jsonl"
        docs = data.load_split(frozen, filename)
        paths = {group: [router.encode(row["text"], row["lang"], group) for row in docs]
                 for group in ("G1", "G2", "G3")}
        digests[split] = {group: path_digest(((row, None, None) for row in docs), ids)
                          for group, ids in paths.items()}
        native = {name: digests[split][group] for group, name in zip(("G2", "G3"), BOUNDARY_GROUPS, strict=True)}
        if native != selection["token_path_sha256"][split]:
            raise ValueError(f"{split}: LangMAP or G3 IDs differ from the full-split evaluation")
        if split == "test" and (digests[split]["G1"] != bpe_reference["g1_test_token_path_sha256"]
                                or data.sha256_file(frozen / filename) != bpe_reference["test_sha256"]):
            raise ValueError("Test BPE IDs differ from the full-split evaluation")

        indices = single_window_indices(paths, context)
        included = set(indices)
        selected = [docs[i] for i in indices]
        total_docs = language_totals(docs, (1 for _ in docs))
        retained_docs = language_totals(selected, (1 for _ in selected))
        total_bytes = language_totals(docs, (len(row["text"].encode("utf-8")) for row in docs))
        retained_bytes = language_totals(selected, (len(row["text"].encode("utf-8")) for row in selected))
        if any(not retained_bytes[lang] for lang in LANGUAGES):
            raise ValueError(f"{split}: no single-window documents for at least one language")
        samples.extend({"split": split, "lang": lang, "documents": retained_docs[lang],
                        "bytes": retained_bytes[lang], "total_documents": total_docs[lang],
                        "total_bytes": total_bytes[lang]} for lang in LANGUAGES)
        audit.extend({"split": split, "lang": row["lang"], "id": row["id"],
                      "bytes": len(row["text"].encode("utf-8")), "included": i in included,
                      **{f"{group}_tokens": len(ids[i]) for group, ids in paths.items()}}
                     for i, row in enumerate(docs))
        prepared[split] = (selected, {group: [(docs[i]["lang"], ids[i]) for i in indices]
                                     for group, ids in paths.items()})
    return prepared, samples, audit, digests


def score_group(model, group, route, prepared, pilot, device, seed):
    rows = []
    for split, (docs, paths) in prepared.items():
        print(f"Scoring {split} {group}, seed {seed}, {len(docs)} documents", flush=True)
        scores = train.evaluate(model, docs, paths[route], pilot["eos_id"], pilot["context"], device)
        rows.extend({"seed": seed, "split": split, "group": group, **score} for score in scores)
    return rows


def paired_contrasts(rows, seeds):
    lookup = {(row["seed"], row["split"], row["lang"], row["group"]): row for row in rows}
    paired, summary = [], []
    for split, lang, (contrast, (left, right)) in product(SPLITS, LANGUAGES, CONTRASTS.items()):
        compared_seeds = (None,) if left in BASE_ROUTES else seeds
        values = []
        for seed in compared_seeds:
            a, b = (lookup[seed, split, lang, group] for group in (left, right))
            if a["bytes"] != b["bytes"]:
                raise ValueError(f"Unpaired corpus bytes for {split}/{lang}/{contrast}")
            delta = a["bpb"] - b["bpb"]
            values.append(delta)
            paired.append({"seed": seed, "split": split, "lang": lang, "contrast": contrast,
                           "bytes": a["bytes"], "bpb_difference": delta})
        summary.append({"split": split, "lang": lang, "contrast": contrast, "bpb_difference_mean": mean(values),
                        "bpb_difference_sample_sd": stdev(values) if len(values) > 1 else "",
                        "observations": len(values)})
    return paired, summary


def score(run, out, seeds):
    if out.exists():
        raise FileExistsError(f"{out} already exists. Choose another --out to rescore.")

    import torch
    from transformers import set_seed

    pilot = data.read_json(run / "seed-42/run.json")
    if pilot["context"] != 512:
        raise ValueError("Expected the frozen 512-position pilot window")
    device, dtype = verified_runtime(pilot)
    router = SameRowsModels(run, run / "g3-eval/review-g2-same-rows", device=device, dtype=dtype)
    prepared, samples, audit, digests = prepare_splits(run, router, pilot["context"])
    checkpoints = {"base": router.provenance("G0"), **{
        str(seed): {group: router.provenance(group, seed) for group in ROUTES} for seed in seeds
    }}
    set_seed(42)
    model = router.load("G0")
    rows = [row for group, route in BASE_ROUTES.items()
            for row in score_group(model, group, route, prepared, pilot, device, None)]
    del model
    torch.cuda.empty_cache()
    for seed, (group, route) in product(seeds, ROUTES.items()):
        set_seed(seed)
        model = router.load(group, seed)
        rows.extend(score_group(model, group, route, prepared, pilot, device, seed))
        del model
        torch.cuda.empty_cache()

    paired, summary = paired_contrasts(rows, seeds)
    manifest = {
        **{key: pilot[key] for key in ("model", "model_revision", "device", "cuda", "dtype", "packages", "context", "eos_id")},
        "seeds": seeds,
        "references": {str(file.relative_to(run)): data.sha256_file(file) for file in (
            run / "g3-eval/manifest.json", run / "g3-eval/boundary-pilot/run.json", run / "g3-eval/review-bpe/run.json",
        )},
        "eligibility": "1 to 511 content tokens on all BPE, native LangMAP and morphology-guided paths",
        "full_token_paths_sha256": digests,
        "checkpoints": checkpoints,
        "scoring_code_sha256": data.sha256_file(Path(train.__file__)),
        "window_code_sha256": data.sha256_file(Path(data.__file__)),
        "analysis_code_sha256": data.sha256_file(Path(__file__)),
    }
    publish(out, manifest, {"subset.csv": samples, "documents.csv": audit, "bpb.csv": rows,
                            "paired.csv": paired, "summary.csv": summary})
    print(f"Single-window FineWeb results: {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/pilot"))
    parser.add_argument("--out", type=Path, help="defaults to RUN/g3-eval/single-window-bpb")
    parser.add_argument("--seed", type=int, nargs="+", choices=SEEDS, default=list(SEEDS))
    args = parser.parse_args()
    if len(args.seed) != len(set(args.seed)):
        parser.error("each --seed value must be unique")
    score(args.run, args.out or args.run / "g3-eval/single-window-bpb", args.seed)


if __name__ == "__main__":
    main()
