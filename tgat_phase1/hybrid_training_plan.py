"""Freeze the known-language training route while reusing the original G4 rows."""

from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .data import LANGUAGES, load_split, read_json, sha256_file, write_json
from .embedding_ablation import SEEDS, inputs as embedding_inputs, read_plan as read_embedding_plan
from .g3_lora import budget, check_budget
from .row_selection import _row_digest, _update_digest
from .tokenization import encode_documents


ROUTING = {"eng_Latn": "original-bpe", "jpn_Jpan": "original-bpe", "tur_Latn": "g3-boundary"}


def _source_inputs(run):
    boundary, _, pilot = embedding_inputs(run)
    embedding = run / "g3-eval" / "embedding-ablation"
    source = read_embedding_plan(embedding, boundary, pilot)
    tokenizer = run / "langmap" / "model_tokenizer" / "tokenizer.json"
    selected = {}
    for seed in SEEDS:
        draw = source["draws"][str(seed)]
        rows = draw["selected_ids"]
        if (len(rows) != source["size"] or rows != sorted(set(rows)) or pilot["eos_id"] in rows
                or draw["sha256"]["selected_ids"] != _row_digest(rows)):
            raise ValueError(f"Original G4 seed {seed} row IDs differ from their frozen fingerprint")
        selected[str(seed)] = rows
    first = source["draws"][str(SEEDS[0])]["sha256"]
    if (any(rows != next(iter(selected.values())) for rows in selected.values())
            or any(any(source["draws"][str(seed)]["sha256"][key] != first[key]
                       for key in ("original_paths", "target_paths")) for seed in SEEDS)):
        raise ValueError("Frozen G4 seeds do not share a single row set and tokenizer paths")
    identity = {
        "schema_version": 1, "routing": ROUTING, "seeds": list(SEEDS), "pilot": pilot,
        "row_ids_by_seed": selected, "original_g4_training_budget": source["training_budget"],
        "source_embedding_plan_sha256": sha256_file(embedding / "run.json"),
        "source_g3_training_ids_sha256": source["training_ids_sha256"],
        "source_original_tokenizer_sha256": sha256_file(tokenizer),
        "source_g3_selection_sha256": source["selection_sha256"],
        "source_original_paths_sha256": first["original_paths"],
        "source_g3_paths_sha256": first["target_paths"],
    }
    return identity, tokenizer, embedding


def _verify_paths(paths, expected, label):
    digest = hashlib.sha256()
    for ids in paths:
        _update_digest(digest, ids)
    if digest.hexdigest() != expected:
        raise ValueError(f"{label} token IDs differ from the frozen G4 plan")


def _training_stats(docs, paths, context, row_ids):
    counts = {lang: Counter() for lang in LANGUAGES}
    by_language = {
        lang: {"documents": 0, "bytes": 0, "train_tokens": 0, "causal_windows": 0, "input_tokens": 0}
        for lang in LANGUAGES
    }
    for doc, ids in zip(docs, paths, strict=True):
        lang = doc["lang"]
        stats = by_language[lang]
        stats["documents"] += 1
        stats["bytes"] += len(doc["text"].encode("utf-8"))
        stats["train_tokens"] += len(ids)
        stats["causal_windows"] += (len(ids) + context - 2) // (context - 1)
        # Each window's last token becomes the next window's context. Only the
        # final token of a document has no outgoing loss. EOS is never selected.
        stats["input_tokens"] += len(ids) - 1
        counts[lang].update(ids[:-1])

    def support(rows):
        tracked = {lang: [counts[lang][token] for token in rows] for lang in LANGUAGES}

        def describe(occurrences, total):
            observed = sum(count > 0 for count in occurrences)
            row_inputs = sum(occurrences)
            return {"input_tokens": total, "selected_input_tokens": row_inputs, "observed_rows": observed,
                    "unobserved_rows": len(rows) - observed,
                    "selected_input_fraction": row_inputs / total if total else 0.0}

        total_inputs = sum(stats["input_tokens"] for stats in by_language.values())
        across_languages = [sum(values) for values in zip(*tracked.values(), strict=True)]
        return {
            **describe(across_languages, total_inputs),
            "selected_rows": len(rows),
            "per_row_input_counts": across_languages,
            "per_language": {
                lang: {**describe(tracked[lang], by_language[lang]["input_tokens"]),
                       "per_row_input_counts": tracked[lang]}
                for lang in LANGUAGES
            },
        }

    return by_language, {seed: support(rows) for seed, rows in row_ids.items()}


def _read_plan(out, identity):
    manifest = read_json(out / "plan.json")
    if any(manifest.get(key) != value for key, value in identity.items()):
        raise ValueError("Hybrid training plan differs from the original frozen G4 inputs")
    if sha256_file(out / "training_ids.json") != manifest["training_ids_sha256"]:
        raise ValueError("Hybrid training token paths failed SHA-256 verification")
    return manifest


def read_plan(run: Path, out: Path):
    """Read and verify the plan against the archived all-G3 G4 inputs."""
    identity, _, _ = _source_inputs(run)
    return _read_plan(out, identity)


def prepare(run: Path, out: Path):
    """Reuse cached G3 Turkish paths and the original BPE paths for English and Japanese."""
    identity, tokenizer_file, embedding = _source_inputs(run)
    if out.exists():
        return _read_plan(out, identity)

    from tokenizers import Tokenizer

    pilot = identity["pilot"]
    if version("tokenizers") != pilot["packages"]["tokenizers"]:
        raise ValueError("Hybrid planning requires the pilot's pinned Tokenizers runtime")
    docs = load_split(run / "corpus", "train.jsonl")
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    if tokenizer.get_vocab_size() != pilot["tokenizer_vocab_entries"]:
        raise ValueError("Saved original BPE differs from the pilot vocabulary")
    bpe = [ids for _, ids in encode_documents(docs, tokenizer)]
    g3 = read_json(embedding / "training_ids.json")
    _verify_paths(bpe, identity["source_original_paths_sha256"], "Original BPE")
    _verify_paths(g3, identity["source_g3_paths_sha256"], "Cached G3")
    context, accumulation = pilot["context"], pilot["grad_accum"]
    original_budget = budget(bpe, context, accumulation)
    check_budget(read_json(run / "seed-42" / "G1-original" / "training.json"), original_budget, "Original BPE")
    check_budget(identity["original_g4_training_budget"], budget(g3, context, accumulation), "Original G4")
    paths = [original if ROUTING[doc["lang"]] == "original-bpe" else boundary
             for doc, original, boundary in zip(docs, bpe, g3, strict=True)]
    language_stats, row_support = _training_stats(docs, paths, context, identity["row_ids_by_seed"])
    original_stats, original_row_support = _training_stats(docs, g3, context, identity["row_ids_by_seed"])

    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".hybrid-training-", dir=out.parent) as temporary:
        staged = Path(temporary) / "plan"
        staged.mkdir()
        token_file = staged / "training_ids.json"
        with token_file.open("w", encoding="utf-8") as stream:
            json.dump(paths, stream, separators=(",", ":"))
        manifest = {
            **identity,
            "training_budget": budget(paths, context, accumulation),
            "original_bpe_training_budget": original_budget,
            "per_language": language_stats,
            "original_g4_per_language": original_stats,
            "row_input_support_by_seed": row_support,
            "original_g4_row_input_support_by_seed": original_row_support,
            "row_input_support_definition": "preceding content token with an outgoing loss, excluding EOS",
            "training_ids_sha256": sha256_file(token_file),
        }
        write_json(staged / "plan.json", manifest)
        staged.rename(out)
    return manifest
