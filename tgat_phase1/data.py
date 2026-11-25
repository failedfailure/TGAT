"""Prepare one reproducible corpus shared by all Phase 1 groups."""

import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

LANGUAGES = ("eng_Latn", "jpn_Jpan", "tur_Latn")
MODEL = "Qwen/Qwen3-0.6B-Base"
MODEL_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
LANGMAP_REVISION = "ec6d1db61f5c1ed374717b1c88af2db54235655d"
SOURCES = {
    "eng_Latn": ("HuggingFaceFW/fineweb", "sample-10BT"),
    "jpn_Jpan": ("HuggingFaceFW/fineweb-2", "jpn_Jpan"),
    "tur_Latn": ("HuggingFaceFW/fineweb-2", "tur_Latn"),
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def canonical_text(text):
    if not isinstance(text, str):
        return ""
    # Qwen3's tokenizer normalizes to NFC. Use the same bytes for both groups.
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def document_splits(sources, train_docs, eval_docs, seed, *, exclude=()):
    """Select distinct documents, then hold out a seeded subset per language."""
    seen = set(exclude)
    train, heldout = [], []
    for lang in LANGUAGES:
        selected = []
        for row in sources[lang]:
            text = canonical_text(row.get("text"))
            encoded = text.encode("utf-8")
            if not text.strip() or len(encoded) < 24 or len(encoded) > 16000:
                continue
            # Avoid interpreting literal Qwen control markers as corpus content.
            if re.search(r"<\|[^|\n]{1,80}\|>", text):
                continue
            identity = hashlib.sha256(encoded).hexdigest()
            if identity in seen:
                continue
            seen.add(identity)
            selected.append({"id": identity, "lang": lang, "text": text})
            if len(selected) == train_docs + eval_docs:
                break
        if len(selected) != train_docs + eval_docs:
            raise ValueError(f"{lang}: only {len(selected)} usable documents, "
                             f"need {train_docs + eval_docs}")
        random.Random(f"{seed}:{lang}").shuffle(selected)
        heldout.extend(selected[:eval_docs])
        train.extend(selected[eval_docs:])
    return train, heldout


def split_records(records, start_id, context):
    """Make causal windows; a boundary token supplies context exactly once."""
    for lang, ids in records:
        for start in range(0, len(ids), context - 1):
            previous = ids[start - 1] if start else start_id
            yield lang, [previous, *ids[start:start + context - 1]]


def load_split(corpus, filename):
    manifest = read_json(corpus / "manifest.json")
    path = corpus / filename
    if sha256_file(path) != manifest["splits"][filename]["sha256"]:
        raise ValueError(f"Corpus checksum mismatch: {path}")
    return list(read_jsonl(path))


def prepare(args):
    out = args.run / "corpus"
    if out.exists():
        raise FileExistsError(f"{out} exists; choose a new run or reuse the completed corpus")
    if args.local_data:
        paths = {lang: args.local_data / f"{lang}.jsonl" for lang in LANGUAGES}
        streams = {lang: read_jsonl(path) for lang, path in paths.items()}
        provenance = {lang: {"file": str(path), "sha256": sha256_file(path)}
                      for lang, path in paths.items()}
    else:
        from datasets import load_dataset
        from huggingface_hub import HfApi

        api = HfApi()
        revisions = {name: api.dataset_info(name).sha for name in sorted({v[0] for v in SOURCES.values()})}
        streams = {lang: load_dataset(name, config, split="train", streaming=True,
                                      revision=revisions[name])
                   for lang, (name, config) in SOURCES.items()}
        provenance = {lang: {"repository": name, "config": config, "revision": revisions[name]}
                      for lang, (name, config) in SOURCES.items()}
    train, heldout = document_splits(streams, args.train_docs, args.eval_docs, args.split_seed)
    write_splits(out, {"sources": provenance, "split_seed": args.split_seed},
                 {"train.jsonl": train, "heldout.jsonl": heldout})


def write_splits(out, manifest, splits):
    out.mkdir(parents=True)
    manifest = {**manifest, "splits": {}}
    for filename, documents in splits.items():
        path = out / filename
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in documents:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["splits"][filename] = {
            "documents": len(documents),
            "bytes": sum(len(row["text"].encode("utf-8")) for row in documents),
            "sha256": sha256_file(path),
        }
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest["splits"], indent=2))


def prepare_g3(args):
    pilot_corpus = args.run / "corpus"
    out = args.run / "g3-eval"
    if out.exists():
        raise FileExistsError(f"{out} already exists")
    manifest = read_json(pilot_corpus / "manifest.json")
    pilot_docs = (load_split(pilot_corpus, "train.jsonl")
                  + load_split(pilot_corpus, "heldout.jsonl"))
    excluded = {row["id"] for row in pilot_docs}
    sources = manifest["sources"]
    from datasets import load_dataset

    streams = {lang: load_dataset(source["repository"], source["config"],
                                  revision=source["revision"], split="train", streaming=True)
               for lang, source in sources.items()}
    test, validation = document_splits(streams, args.test_docs, args.validation_docs,
                                       args.split_seed, exclude=excluded)
    write_splits(out, {"sources": sources, "split_seed": args.split_seed,
                       "excluded_corpus_manifest_sha256": sha256_file(pilot_corpus / "manifest.json"),
                       "excluded_document_count": len(excluded)},
                 {"validation.jsonl": validation, "test.jsonl": test})
