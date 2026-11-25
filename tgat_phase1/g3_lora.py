"""Compare one matched G3 LoRA run with an existing G2 adapter."""

import argparse
from collections import Counter
import csv
import fcntl
from importlib.metadata import version
from itertools import product
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from .boundaries import training_stems
from .data import LANGUAGES, MODEL, MODEL_REVISION, load_split, read_json, sha256_file, write_json
from .g3 import (
    GROUPS, SPLITS, fineweb_paths, load_selection, native_langmap, prepare, publish,
    verified_eval_paths, verify_native,
)
from .rescoring import load_segmenters
from .series import inspect_seed
from .tokenization import encode_documents
from .train import evaluate, hardware, load_model, tied_weights, train_lora


def language_totals(docs, amounts):
    totals = Counter()
    for doc, amount in zip(docs, amounts, strict=True):
        totals[doc["lang"]] += amount
    totals["all"] = sum(totals.values())
    return totals


def verified_control(run, seed, selection):
    pilot = read_json(run / "seed-42/run.json")
    expected = {"model": MODEL, "model_revision": MODEL_REVISION, "epochs": 1, "microbatch": 1,
                "corpus_manifest_sha256": selection["inputs"]["corpus"],
                "langmap_manifest_sha256": selection["inputs"]["langmap"]}
    if any(pilot.get(key) != value for key, value in expected.items()):
        raise ValueError("G3 needs the frozen one-epoch pilot configuration")
    heldout = load_split(run / "corpus", "heldout.jsonl")
    counts = language_totals(heldout, (len(row["text"].encode("utf-8")) for row in heldout))
    status, detail, _ = inspect_seed(run, seed, pilot, counts)
    if status != "verified":
        raise ValueError(f"G3 needs a verified G2 seed {seed}: {detail}")
    return {**pilot, "seed": seed}, heldout, run / f"seed-{seed}" / GROUPS[0]


def verified_base(out, selection, pilot, eval_splits):
    directory = out / "base-bpb"
    manifest = read_json(directory / "run.json")
    expected = dict(
        model=MODEL, model_revision=MODEL_REVISION, context=pilot["context"], eos_id=pilot["eos_id"],
        dtype=pilot["dtype"], selected_lambda=selection["selected_lambda"], native_g2_ids_verified=True,
        selection_manifest_sha256=sha256_file(out / "run.json"),
        packages={key: pilot["packages"][key] for key in ("torch", "transformers", "tokenizers")},
    )
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Base BPB does not match the selected G3 paths and one-epoch pilot")
    languages = (*LANGUAGES, "all")
    expected_metrics = {}
    for split, (docs, paths) in eval_splits.items():
        sizes = language_totals(docs, (len(row["text"].encode("utf-8")) for row in docs))
        for group, token_ids in paths.items():
            tokens = language_totals(docs, map(len, token_ids))
            expected_metrics.update({(split, group, lang): (sizes[lang], tokens[lang]) for lang in languages})
    scores = {}
    with (directory / "bpb.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = row["split"], row["group"], row["lang"]
            size, tokens = int(row["bytes"]), int(row["tokens"])
            nll, bpb = float(row["nll_nats"]), float(row["bpb"])
            if (key in scores or expected_metrics.get(key) != (size, tokens)
                    or not math.isfinite(nll) or nll <= 0
                    or not math.isclose(bpb, nll / (size * math.log(2)), rel_tol=1e-9)):
                raise ValueError(f"Invalid base BPB row {key}")
            scores[key] = {"stage": "before_lora", **row, "bytes": size, "tokens": tokens,
                           "nll_nats": nll, "bpb": bpb}
    if scores.keys() != expected_metrics.keys():
        raise ValueError("Base BPB is missing rows")
    for split, group in product(SPLITS, GROUPS):
        if not math.isclose(scores[split, group, "all"]["nll_nats"],
                            sum(scores[split, group, lang]["nll_nats"] for lang in LANGUAGES)):
            raise ValueError(f"Invalid base BPB aggregate for {split} {group}")
    return manifest, list(scores.values())


def budget(ids_by_doc, context, accumulation):
    targets = sum(map(len, ids_by_doc))
    windows = sum((len(ids) + context - 2) // (context - 1) for ids in ids_by_doc)
    return {"train_tokens": targets, "optimizer_steps": (windows + accumulation - 1) // accumulation}


def check_budget(stats, expected, group):
    if any(stats[key] != value for key, value in expected.items()):
        raise ValueError(f"{group} training counters differ from the planned windows")


def saved_model(base, directory):
    from peft import PeftModel

    model = PeftModel.from_pretrained(base, str(directory / "adapter"), is_trainable=False)
    weights = (model.get_input_embeddings().weight, model.get_output_embeddings().weight)
    if not tied_weights(model) or any(weight.requires_grad for weight in weights):
        raise ValueError(f"{directory}: expected frozen, tied embedding and output weights")
    return model


def verify_reloaded_control(model, control, heldout, native, pilot, device):
    reference = {row["lang"]: row for row in read_json(control / "heldout.json")["after_lora"]}
    measured = evaluate(model, heldout, list(encode_documents(heldout, native)), pilot["eos_id"],
                        pilot["context"], device)
    for row in measured:
        expected = reference[row["lang"]]
        if (any(row[key] != expected[key] for key in ("bytes", "tokens"))
                or not math.isclose(row["bpb"], expected["bpb"], rel_tol=0.0, abs_tol=0.001)):
            raise ValueError(f"Reloaded G2 differs from the original pilot holdout for {row['lang']}")


def evaluate_group(model, eval_splits, group, pilot, device):
    results = []
    for split, (docs, paths) in eval_splits.items():
        print(f"Scoring {split} {group} after LoRA", flush=True)
        encoded = [(row["lang"], ids) for row, ids in zip(docs, paths[group], strict=True)]
        measured = evaluate(model, docs, encoded, pilot["eos_id"], pilot["context"], device)
        results.extend({"stage": "after_lora", "split": split, "group": group, **row} for row in measured)
    return results


def g3_adapter(base, directory, docs, ids, pilot, expected, device):
    from transformers import set_seed

    if directory.exists():
        check_budget(read_json(directory / "training.json"), expected, GROUPS[1])
        print(f"Reusing trained G3 adapter in {directory}", flush=True)
        return saved_model(base, directory)
    options = argparse.Namespace(seed=pilot["seed"], epochs=pilot["epochs"], context=pilot["context"],
                                 grad_accum=pilot["grad_accum"], learning_rate=pilot["learning_rate"],
                                 no_gradient_checkpointing=not pilot["gradient_checkpointing"])
    set_seed(pilot["seed"])
    print(f"Training {GROUPS[1]} for one epoch, seed {pilot['seed']}", flush=True)
    encoded = [(row["lang"], path) for row, path in zip(docs, ids, strict=True)]
    model, stats = train_lora(base, encoded, pilot["eos_id"], options, device)
    check_budget(stats, expected, GROUPS[1])
    with TemporaryDirectory(prefix=".adapter-", dir=directory.parent) as temporary:
        saved = Path(temporary) / "trained"
        saved.mkdir()
        model.save_pretrained(saved / "adapter", save_embedding_layers=False)
        write_json(saved / "training.json", stats)
        saved.rename(directory)
    return model


def report_gaps(rows):
    scores = {(row["stage"], row["split"], row["group"], row["lang"]): row["bpb"] for row in rows}
    print("FineWeb test    G2 before  G3 before   G2 after   G3 after    Gap change")
    for lang in (*LANGUAGES, "all"):
        a, b, c, d = (scores[stage, "test", group, lang]
                      for stage, group in product(("before_lora", "after_lora"), GROUPS))
        print(f"{lang:12}    {a:9.4f}  {b:9.4f}  {c:9.4f}  {d:9.4f}    {(d-c)-(b-a):+10.4f}")


def compare(run, reference, out, args):
    import torch
    from transformers import set_seed

    directory = out / f"lora-seed-{args.seed}"
    selection = load_selection(run, reference, out)
    with (out / f".lora-seed-{args.seed}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"G3 LoRA seed {args.seed} is already running") from None
        if (directory / "bpb.csv").exists():
            print(f"Results already saved: {directory / 'bpb.csv'}")
            return
        pilot, heldout, control = verified_control(run, args.seed, selection)
        device, dtype = hardware(args)
        packages = {name: version(name) for name in pilot["packages"]}
        if packages != pilot["packages"] or str(dtype) != pilot["dtype"] or torch.version.cuda != pilot["cuda"]:
            raise ValueError("G3 LoRA needs the same runtime versions and precision as the verified G2 pilot")
        train = load_split(run / "corpus", "train.jsonl")
        stems, segmenters, native = training_stems(train), load_segmenters(run), native_langmap(run)
        eval_splits = verified_eval_paths(run, selection, segmenters, stems, native)
        base, results = verified_base(out, selection, pilot, eval_splits)
        if str(device) != base["device"]:
            raise ValueError("G3 LoRA and base BPB must use the same device type")
        prepared = prepare(train, segmenters, stems)
        paths = fineweb_paths(prepared, segmenters, selection["selected_lambda"])
        verify_native(native, train, paths[GROUPS[0]])
        budgets = {group: budget(ids, pilot["context"], pilot["grad_accum"]) for group, ids in paths.items()}
        check_budget(read_json(control / "training.json"), budgets[GROUPS[0]], GROUPS[0])
        manifest = {**pilot, "device": torch.cuda.get_device_name(device),
                    "selected_lambda": selection["selected_lambda"], "length_weight": selection["length_weight"],
                    "training_budgets": budgets, "selection_sha256": sha256_file(out / "run.json"),
                    "base_bpb_sha256": sha256_file(out / "base-bpb/bpb.csv"),
                    "g2_adapter_sha256": sha256_file(control / "adapter/adapter_model.safetensors"),
                    "g2_adapter_config_sha256": sha256_file(control / "adapter/adapter_config.json")}
        if directory.exists():
            if read_json(directory / "run.json") != manifest:
                raise ValueError(f"{directory}: saved run inputs differ from this attempt")
        else:
            publish(directory, manifest, {})
        print(f"Matched training budgets: {budgets}", flush=True)
        set_seed(args.seed)
        base_model = load_model(device, dtype)
        g2 = saved_model(base_model, control)
        verify_reloaded_control(g2, control, heldout, native, pilot, device)
        results.extend(evaluate_group(g2, eval_splits, GROUPS[0], pilot, device))
        del g2, base_model
        torch.cuda.empty_cache()
        set_seed(args.seed)
        base_model = load_model(device, dtype)
        g3 = g3_adapter(base_model, directory / GROUPS[1], train, paths[GROUPS[1]], pilot, budgets[GROUPS[1]], device)
        results.extend(evaluate_group(g3, eval_splits, GROUPS[1], pilot, device))
        pending = directory / "bpb.csv.tmp"
        with pending.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        pending.replace(directory / "bpb.csv")
        report_gaps(results)
        print(f"Results: {directory / 'bpb.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/pilot"))
    parser.add_argument("--reference", type=Path, help="override the frozen stem-boundary reference")
    parser.add_argument("--out", type=Path, help="override the boundary pilot result directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    g3 = args.run / "g3-eval"
    args.device = "cuda"
    compare(args.run, args.reference or g3 / "stem-reference",
            args.out or g3 / "boundary-pilot", args)


if __name__ == "__main__":
    main()
