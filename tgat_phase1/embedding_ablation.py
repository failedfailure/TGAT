"""Run the matched selective-embedding comparison on the frozen G3 paths."""

import argparse
import csv
import fcntl
from importlib.metadata import version
from itertools import product
import json
import math
from pathlib import Path
from statistics import mean, stdev
from tempfile import TemporaryDirectory

from . import row_selection
from .boundaries import training_stems
from .data import LANGUAGES, load_split, read_json, sha256_file, write_json
from .g3 import (
    SPLITS, fineweb_paths, load_selection, native_langmap, prepare, verified_eval_paths, verify_native,
)
from .g3_lora import budget, check_budget, evaluate_group, saved_model, verified_control
from .rescoring import load_segmenters
from .tokenization import encode_documents, original_tokenizer
from .train import hardware, load_model, train_lora

SEEDS = (42, 43, 44)
CONTROL = "G3-boundary"
GROUPS = {"G4-drift": "selected_ids", "G6-random": "random_ids"}
FULL_GROUP = "G5-full"
LANGS = (*LANGUAGES, "all")


def inputs(run):
    frozen = run / "g3-eval"
    boundary = frozen / "boundary-pilot"
    selection = load_selection(run, frozen / "stem-reference", boundary)
    pilot, _, _ = verified_control(run, SEEDS[0], selection)
    return boundary, selection, pilot


def plan_identity(boundary, pilot, size, min_support):
    return {
        "selection_sha256": sha256_file(boundary / "run.json"),
        "model_revision": pilot["model_revision"],
        "tokenizer_packages": {name: pilot["packages"][name] for name in ("transformers", "tokenizers")},
        "seeds": list(SEEDS),
        "size": size,
        "min_support": min_support,
        "row_selection_code_sha256": sha256_file(Path(row_selection.__file__)),
    }


def read_plan(out, boundary, pilot):
    plan = read_json(out / "run.json")
    expected = plan_identity(boundary, pilot, plan["size"], plan["min_support"])
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError("Embedding plan differs from the frozen G3 inputs or selection code")
    if sha256_file(out / "training_ids.json") != plan["training_ids_sha256"]:
        raise ValueError("Embedding plan training paths failed the checksum")
    return plan


def make_plan(run, out, size, min_support):
    boundary, selection, pilot = inputs(run)
    identity = plan_identity(boundary, pilot, size, min_support)
    if out.exists():
        saved = read_plan(out, boundary, pilot)
        if any(saved.get(k) != v for k, v in identity.items()):
            raise ValueError("Existing plan has different settings. Use another --out directory.")
        print(f"Verified embedding plan: {out / 'run.json'}")
        return
    if any(version(name) != pinned for name, pinned in identity["tokenizer_packages"].items()):
        raise ValueError("Row planning requires the pilot's pinned tokenizer runtime")
    train = load_split(run / "corpus", "train.jsonl")
    tokenizer = original_tokenizer()
    if tokenizer.eos_token_id != pilot["eos_id"]:
        raise ValueError("Original tokenizer EOS differs from the frozen pilot")
    original = [ids for _, ids in encode_documents(train, tokenizer.backend_tokenizer)]
    vocab = tokenizer.backend_tokenizer.get_vocab()
    if len(vocab) != pilot["tokenizer_vocab_entries"]:
        raise ValueError("Original tokenizer vocab differs from the frozen pilot")
    special = set(tokenizer.all_special_ids)
    special.update(i for i, token in tokenizer.added_tokens_decoder.items() if token.special)
    ordinary = set(vocab.values()) - special
    check_budget(read_json(run / "seed-42/G1-original/training.json"),
                 budget(original, pilot["context"], pilot["grad_accum"]), "Original BPE")
    print("Scoring frozen G3 training paths for drift selection", flush=True)
    segmenters, stems = load_segmenters(run), training_stems(train)
    prepared = prepare(train, segmenters, stems)
    paths = fineweb_paths(prepared, segmenters, selection["selected_lambda"])
    verify_native(native_langmap(run), train, paths["G2-langmap"])
    g3_ids = paths[CONTROL]
    expected = budget(g3_ids, pilot["context"], pilot["grad_accum"])
    draws = {}
    for seed in SEEDS:
        control = boundary / f"lora-seed-{seed}"
        check_budget(read_json(control / CONTROL / "training.json"), expected, CONTROL)
        draws[str(seed)] = row_selection.select_rows(
            original, g3_ids, ordinary, size=size, min_support=min_support, seed=seed,
        )
    if any(draw["selected_ids"] != draws[str(SEEDS[0])]["selected_ids"] for draw in draws.values()):
        raise RuntimeError("G4 selection must be identical across seeds")
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".embedding-plan-", dir=out.parent) as temporary:
        staged = Path(temporary) / "plan"
        staged.mkdir()
        encoded = staged / "training_ids.json"
        with encoded.open("w", encoding="utf-8") as stream:
            json.dump(g3_ids, stream, separators=(",", ":"))
        write_json(staged / "run.json", {**identity, "training_budget": expected,
                                        "training_ids_sha256": sha256_file(encoded), "draws": draws})
        staged.rename(out)
    for seed, draw in draws.items():
        print(f"Seed {seed}: {draw['eligible_rows']} eligible rows, "
              f"{size} rows per group, {draw['overlap']} rows overlap")
    print(f"Embedding plan: {out / 'run.json'}")


def read_scores(path, groups, splits):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    scores = {}
    for row in rows:
        key = row["split"], row["group"], row["lang"]
        nll, bpb = float(row["nll_nats"]), float(row["bpb"])
        size, count = int(row["bytes"]), int(row["tokens"])
        if (key in scores or size <= 0 or count <= 0 or not math.isfinite(nll) or nll <= 0
                or not math.isclose(bpb, nll / (size * math.log(2)), rel_tol=1e-9)):
            raise ValueError(f"Invalid score row in {path}: {key}")
        scores[key] = {**row, "bytes": size, "tokens": count, "nll_nats": nll, "bpb": bpb}
    if scores.keys() != set(product(splits, groups, LANGS)):
        raise ValueError(f"Missing or unexpected score rows in {path}")
    for split, lang in product(splits, LANGS):
        matched = {(scores[split, group, lang]["bytes"], scores[split, group, lang]["tokens"])
                   for group in groups}
        if len(matched) != 1:
            raise ValueError(f"Different evaluation text or token counts in {path}: {split}, {lang}")
    return scores


def baseline(boundary, seed, pilot, plan, eval_splits):
    control = boundary / f"lora-seed-{seed}"
    saved = read_json(control / "run.json")
    expected = {**pilot, "seed": seed}
    if (any(saved.get(k) != v for k, v in expected.items())
            or saved["selection_sha256"] != plan["selection_sha256"]
            or saved["training_budgets"][CONTROL] != plan["training_budget"]):
        raise ValueError(f"Existing G3 seed {seed} differs from this experiment")
    check_budget(read_json(control / CONTROL / "training.json"), plan["training_budget"], CONTROL)
    with (control / "bpb.csv").open(newline="", encoding="utf-8") as stream:
        rows = [{**row, "bytes": int(row["bytes"]), "tokens": int(row["tokens"]),
                 "nll_nats": float(row["nll_nats"]), "bpb": float(row["bpb"])}
                for row in csv.DictReader(stream) if row["stage"] == "after_lora" and row["group"] == CONTROL]
    expected_rows = set(product(SPLITS, LANGS))
    if len(rows) != len(expected_rows) or {(row["split"], row["lang"]) for row in rows} != expected_rows:
        raise ValueError(f"Existing G3 seed {seed} has incomplete BPB results")
    for row in rows:
        docs, paths = eval_splits[row["split"]]
        pairs = [(d, ids) for d, ids in zip(docs, paths[CONTROL], strict=True)
                 if row["lang"] in (d["lang"], "all")]
        size = sum(len(d["text"].encode("utf-8")) for d, _ in pairs)
        count = sum(len(ids) for _, ids in pairs)
        if ((row["bytes"], row["tokens"]) != (size, count)
                or not math.isfinite(row["nll_nats"]) or row["nll_nats"] <= 0
                or not math.isclose(row["bpb"], row["nll_nats"] / (size * math.log(2)), rel_tol=1e-9)):
            raise ValueError(f"Existing G3 seed {seed} has an invalid BPB row or different token paths")
    return rows, sha256_file(control / "bpb.csv")


def smoke_items(items, docs, per_language):
    indexes = [i for lang in LANGUAGES
               for i in [j for j, row in enumerate(docs) if row["lang"] == lang][:per_language]]
    return [items[i] for i in indexes]


def artifact_names(group):
    embedding = "embedding.safetensors" if group == FULL_GROUP else "rows.safetensors"
    return "adapter/adapter_config.json", "adapter/adapter_model.safetensors", embedding


def verified_group(directory, expected, smoke, *, smoke_steps=1):
    stats = read_json(directory / "training.json")
    check_budget(stats, {"optimizer_steps": smoke_steps} if smoke else expected, directory.name)
    if (not stats["reload_logits_exact"] or
            {name: sha256_file(directory / name) for name in artifact_names(directory.name)}
            != stats["artifacts_sha256"]):
        raise ValueError(f"{directory.name}: saved weights differ from the verified checkpoint")
    return stats


def trained_group(directory, ids, encoded, pilot, expected, dtype, device, smoke, *, smoke_steps=1):
    import torch
    from transformers import set_seed

    from .tied_rows import attach_tied_rows, load_tied_rows

    full = directory.name == FULL_GROUP
    if full:
        from .full_embedding import attach_full_embedding, load_full_embedding

        attach = attach_full_embedding
    else:
        def attach(model):
            attach_tied_rows(model, ids)

    def probe(model):
        # Exercise adjusted rows in the input and score the complete vocabulary.
        sample = torch.tensor([[pilot["eos_id"], *ids[:3]]], dtype=torch.long, device=device)
        model.eval()
        with torch.inference_mode():
            return model(input_ids=sample, use_cache=False).logits.detach().cpu()

    def reloaded(base, folder):
        model = saved_model(base, folder)
        if full:
            load_full_embedding(model, folder / "embedding.safetensors")
        else:
            load_tied_rows(model, folder / "rows.safetensors", ids)
        return model

    set_seed(pilot["seed"])
    base = load_model(device, dtype)
    if directory.exists():
        verified_group(directory, expected, smoke, smoke_steps=smoke_steps)
        model = reloaded(base, directory)
    else:
        if full:
            initial_weights = base.get_input_embeddings().weight.detach()[ids[:3]].float().cpu()
        options = argparse.Namespace(**pilot,
                                     no_gradient_checkpointing=not pilot["gradient_checkpointing"])
        model, stats = train_lora(base, encoded, pilot["eos_id"], options, device,
                                  prepare_trainable=attach, max_steps=smoke_steps if smoke else None)
        check_budget(stats, {"optimizer_steps": smoke_steps} if smoke else expected, directory.name)
        if full:
            embedding = model.get_input_embeddings()
            updated = embedding.weight[ids[:3]].detach().cpu()
            if torch.equal(updated, initial_weights):
                raise RuntimeError("The sampled full embedding rows did not update")
            stats["embedding_parameters"] = embedding.weight.numel()
        else:
            embedding = model.tied_rows
            if not torch.isfinite(embedding.delta).all() or not torch.count_nonzero(embedding.delta):
                raise RuntimeError(f"{directory.name}: selected row updates are missing or non-finite")
            stats["embedding_parameters"] = embedding.delta.numel()
        initial_probe = probe(model)
        with TemporaryDirectory(prefix=".trained-", dir=directory.parent) as temporary:
            staged = Path(temporary) / "group"
            model.save_pretrained(staged / "adapter", save_embedding_layers=False)
            embedding.save(staged / artifact_names(directory.name)[-1])
            del embedding, model, base
            torch.cuda.empty_cache()
            base = load_model(device, dtype)
            model = reloaded(base, staged)
            torch.testing.assert_close(probe(model), initial_probe, rtol=0, atol=0)
            stats["reload_logits_exact"] = True
            stats["artifacts_sha256"] = {name: sha256_file(staged / name) for name in artifact_names(directory.name)}
            write_json(staged / "training.json", stats)
            staged.rename(directory)
        print(f"Verified exact logits after saving and reloading {directory.name}", flush=True)
    return model


def fit_group(directory, ids, encoded, pilot, expected, eval_splits, dtype, device, smoke, *, smoke_steps=1):
    model = trained_group(directory, ids, encoded, pilot, expected, dtype, device, smoke, smoke_steps=smoke_steps)
    return evaluate_group(model, eval_splits, directory.name, pilot, device)


def verify_matched_groups(directory, expected, smoke):
    fields = ("train_tokens", "optimizer_steps", "trainable_parameters", "embedding_parameters")
    counts = {tuple(stats[field] for field in fields)
              for stats in (verified_group(directory / name, expected, smoke) for name in GROUPS)}
    if len(counts) != 1:
        raise ValueError("G4 and G6 must have matching token, step and parameter budgets")


def verified_runtime(pilot):
    import torch

    device, dtype = hardware(argparse.Namespace(device="cuda"))
    if ({name: version(name) for name in pilot["packages"]} != pilot["packages"]
            or str(dtype) != pilot["dtype"] or torch.version.cuda != pilot["cuda"]
            or torch.cuda.get_device_name(device) != pilot["device"]):
        raise ValueError("Embedding comparison requires the verified G3 training runtime")
    return device, dtype


def compare(run, out, seed, smoke):
    import torch

    boundary, selection, pilot = inputs(run)
    pilot = {**pilot, "seed": seed}
    plan = read_plan(out, boundary, pilot)
    device, dtype = verified_runtime(pilot)
    directory = out / (f"smoke-seed-{seed}" if smoke else f"seed-{seed}")
    with (out / f".{directory.name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"{directory.name} is already running") from None
        train = load_split(run / "corpus", "train.jsonl")
        train_ids = read_json(out / "training_ids.json")
        expected = budget(train_ids, pilot["context"], pilot["grad_accum"])
        if expected != plan["training_budget"]:
            raise ValueError("Training token or optimization step budget differs from the plan")
        encoded = [(row["lang"], ids) for row, ids in zip(train, train_ids, strict=True)]
        eval_splits = verified_eval_paths(run, selection, load_segmenters(run), training_stems(train),
                                          native_langmap(run))
        if smoke:
            encoded = smoke_items(encoded, train, 2)
            docs, paths = eval_splits["validation"]
            eval_splits = {"validation": (smoke_items(docs, docs, 1),
                                          {group: smoke_items(paths[CONTROL], docs, 1) for group in GROUPS})}
            rows, baseline_sha = [], None
        else:
            rows, baseline_sha = baseline(boundary, seed, pilot, plan, eval_splits)
            eval_splits = {split: (docs, {group: paths[CONTROL] for group in GROUPS})
                           for split, (docs, paths) in eval_splits.items()}
        draw = plan["draws"][str(seed)]
        manifest = {**pilot, "smoke": smoke, "planned_training_budget": expected,
                    "plan_sha256": sha256_file(out / "run.json"), "g3_bpb_sha256": baseline_sha,
                    "row_ids": {group: draw[key] for group, key in GROUPS.items()}}
        directory.mkdir(exist_ok=True)
        run_manifest = directory / "run.json"
        if run_manifest.exists():
            if read_json(run_manifest) != manifest:
                raise ValueError(f"{directory}: different run settings")
        else:
            write_json(run_manifest.with_suffix(".tmp"), manifest)
            run_manifest.with_suffix(".tmp").replace(run_manifest)
        results = directory / "bpb.csv"
        groups = tuple(GROUPS) if smoke else (CONTROL, *GROUPS)
        if not results.exists():
            for group, key in GROUPS.items():
                print(f"Running {group}, seed {seed}, {len(draw[key])} tied rows", flush=True)
                rows.extend(fit_group(directory / group, draw[key], encoded, pilot, expected,
                                      eval_splits, dtype, device, smoke))
                torch.cuda.empty_cache()
            verify_matched_groups(directory, expected, smoke)
            pending = results.with_suffix(".tmp")
            with pending.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            read_scores(pending, groups, eval_splits)
            pending.replace(results)
        else:
            read_scores(results, groups, eval_splits)
            verify_matched_groups(directory, expected, smoke)
        print(f"Verified results: {results}")


def summary(out, seeds):
    tables = []
    for seed in seeds:
        directory = out / f"seed-{seed}"
        expected = read_json(directory / "run.json")["planned_training_budget"]
        verify_matched_groups(directory, expected, smoke=False)
        tables.append(read_scores(directory / "bpb.csv", (CONTROL, *GROUPS), SPLITS))
    print(f"Paired BPB deltas for seeds {', '.join(map(str, seeds))}. Lower is better.")
    print("Split         Language    G3 BPB    G4 BPB    G6 BPB    G4-G6            G4-G3")
    for split, lang in product(SPLITS, LANGS):
        g3, g4, g6 = ([t[split, group, lang]["bpb"] for t in tables] for group in (CONTROL, *GROUPS))

        def delta(other):
            differences = [a - b for a, b in zip(g4, other, strict=True)]
            spread = f" ± {stdev(differences):.4f}" if len(differences) > 1 else ""
            return f"{mean(differences):+.4f}{spread}"

        print(f"{split:13} {lang:9} {mean(g3):9.4f} {mean(g4):9.4f} {mean(g6):9.4f} "
              f"{delta(g6):16} {delta(g3)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "smoke", "run", "summary"))
    parser.add_argument("--run", type=Path, default=Path("runs/pilot"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--size", type=int, default=row_selection.DEFAULT_SIZE, help="rows per arm (plan only)")
    parser.add_argument("--min-support", type=int, default=row_selection.DEFAULT_MIN_SUPPORT,
                        help="minimum G3 training count (plan only)")
    parser.add_argument("--seed", type=int, choices=SEEDS, nargs="+")
    args = parser.parse_args()
    out = args.out or args.run / "g3-eval" / "embedding-ablation"
    if args.action == "plan":
        make_plan(args.run, out, args.size, args.min_support)
    elif args.action == "summary":
        summary(out, args.seed or SEEDS)
    else:
        for seed in args.seed or (SEEDS[0],):
            compare(args.run, out, seed, smoke=args.action == "smoke")


if __name__ == "__main__":
    main()
