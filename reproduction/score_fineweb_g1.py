"""Evaluate the saved one-epoch G1 checkpoints on the frozen Section 5 test split."""

import argparse
import csv
from math import isclose, log
from pathlib import Path
from statistics import mean, stdev

from tgat_phase1 import data, train
from tgat_phase1.downstream_models import DownstreamModels
from tgat_phase1.embedding_ablation import SEEDS, verified_runtime
from tgat_phase1.g3 import path_digest, publish
from tgat_phase1.g3_lora import language_totals
from tgat_phase1.tokenization import encode_documents, original_tokenizer


LANGUAGES = (*data.LANGUAGES, "all")


def read_reference(path, group, expected_bytes):
    rows = {}
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if (row["stage"], row["split"], row["group"]) != ("after_lora", "test", group):
                continue
            lang = row["lang"]
            size, nll, bpb = int(row["bytes"]), float(row["nll_nats"]), float(row["bpb"])
            if lang in rows or size != expected_bytes[lang] or not isclose(bpb, nll / (log(2) * size)):
                raise ValueError(f"Mismatched reference score: {path} {lang}")
            rows[lang] = bpb
    if rows.keys() != set(LANGUAGES):
        raise ValueError(f"Incomplete reference scores: {path}")
    return rows


def score(run, out):
    if out.exists():
        raise FileExistsError(f"{out} already exists. Choose another --out to rescore.")

    import torch
    from transformers import set_seed

    pilot = data.read_json(run / "seed-42/run.json")
    device, dtype = verified_runtime(pilot)
    router = DownstreamModels(run, device=device, dtype=dtype)
    test_path = run / "g3-eval/test.jsonl"
    docs = data.load_split(test_path.parent, test_path.name)
    heldout = data.load_split(run / "corpus", "heldout.jsonl")
    tokenizer = original_tokenizer()
    if tokenizer.eos_token_id != pilot["eos_id"]:
        raise ValueError("Mismatched BPE EOS token")
    test_encoded = list(encode_documents(docs, tokenizer.backend_tokenizer))
    heldout_encoded = list(encode_documents(heldout, tokenizer.backend_tokenizer))
    sizes = language_totals(docs, (len(doc["text"].encode("utf-8")) for doc in docs))
    inputs = {}
    results = []
    for seed in SEEDS:
        print(f"Verifying checkpoints and paired references for seed {seed}", flush=True)
        control = run / f"seed-{seed}/G1-original/heldout.json"
        g2 = run / f"g3-eval/boundary-pilot/lora-seed-{seed}/bpb.csv"
        g4 = run / f"g3-eval/embedding-ablation/seed-{seed}/bpb.csv"
        inputs[str(seed)] = {
            **{group: router.provenance(group, seed) for group in ("G1", "G2", "G4")},
            "g1_pilot_heldout_sha256": data.sha256_file(control),
            "g2_fineweb_bpb_sha256": data.sha256_file(g2),
            "g4_fineweb_bpb_sha256": data.sha256_file(g4),
        }
        references = read_reference(g2, "G2-langmap", sizes), read_reference(g4, "G4-drift", sizes)
        set_seed(seed)
        model = router.load("G1", seed)
        print(f"Checking reloaded G1 seed {seed} on the original pilot holdout", flush=True)
        historical = {row["lang"]: row for row in data.read_json(control)["after_lora"]}
        for row in train.evaluate(model, heldout, heldout_encoded, pilot["eos_id"], pilot["context"], device):
            ref = historical[row["lang"]]
            if (any(row[key] != ref[key] for key in ("bytes", "tokens"))
                    or not isclose(row["bpb"], ref["bpb"], abs_tol=0.001)):
                raise ValueError(f"Reloaded G1 seed {seed} differs on the original pilot holdout: {row['lang']}")
        print(f"Scoring G1 seed {seed} on {len(docs)} frozen FineWeb documents", flush=True)
        for row in train.evaluate(model, docs, test_encoded, pilot["eos_id"], pilot["context"], device):
            lang, g1_bpb = row["lang"], row["bpb"]
            if row["bytes"] != sizes[lang]:
                raise ValueError(f"G1 byte count differs for {lang}")
            g2_bpb, g4_bpb = (reference[lang] for reference in references)
            results.append({"seed": seed, "lang": lang, "test_bytes": row["bytes"],
                            "g1_tokens": row["tokens"], "g1_nll_nats": row["nll_nats"],
                            "g1_bpb": g1_bpb, "g2_bpb": g2_bpb, "g4_bpb": g4_bpb,
                            "g2_minus_g1": g2_bpb - g1_bpb, "g4_minus_g1": g4_bpb - g1_bpb,
                            "g4_minus_g2": g4_bpb - g2_bpb})
        print(results[-1], flush=True)
        del model
        torch.cuda.empty_cache()

    fields = ("g1_bpb", "g2_bpb", "g4_bpb", "g2_minus_g1", "g4_minus_g1", "g4_minus_g2")
    summary = []
    for lang in LANGUAGES:
        paired = [row for row in results if row["lang"] == lang]
        stats = {f"{field}_{name}": agg(row[field] for row in paired) for field in fields
                 for name, agg in (("mean", mean), ("sample_sd", stdev))}
        summary.append({"lang": lang, "test_bytes": sizes[lang], "seeds": len(SEEDS), **stats})
        print(summary[-1], flush=True)
    manifest = {
        "model": pilot["model"], "model_revision": pilot["model_revision"], "seeds": SEEDS,
        "device": pilot["device"], "cuda": pilot["cuda"], "dtype": pilot["dtype"], "packages": pilot["packages"],
        "context": pilot["context"], "eos_id": pilot["eos_id"], "test_documents": len(docs),
        "test_sha256": data.sha256_file(test_path),
        "g1_test_token_path_sha256": path_digest(((doc, None, None) for doc in docs),
                                                 (ids for _, ids in test_encoded)),
        "scoring_code_sha256": data.sha256_file(Path(train.__file__)),
        "window_code_sha256": data.sha256_file(Path(data.__file__)),
        "checkpoints": inputs,
    }
    publish(out, manifest, {"bpb.csv": results, "summary.csv": summary})
    print(f"Results verified: {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/pilot"))
    parser.add_argument("--out", type=Path, help="defaults to RUN/g3-eval/review-bpe")
    args = parser.parse_args()
    score(args.run, args.out or args.run / "g3-eval/review-bpe")


if __name__ == "__main__":
    main()
