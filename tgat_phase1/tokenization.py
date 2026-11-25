"""Fit upstream LangMAP and verify its boundary with the Qwen checkpoint."""

import logging
from time import perf_counter

from .data import (
    LANGMAP_REVISION, LANGUAGES, MODEL, MODEL_REVISION, load_split,
    read_json, sha256_file, write_json,
)


def original_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, use_fast=True)
    if not tokenizer.is_fast or tokenizer.eos_token_id is None:
        raise ValueError("Expected a fast Qwen tokenizer with an EOS ID")
    return tokenizer


def langmap_paths(directory):
    return {lang: directory / f"langspec_{lang}.json" for lang in LANGUAGES}


def checked_langmap(directory, original):
    from tokenizers import Tokenizer
    import json

    expected = original.backend_tokenizer.get_vocab()
    config = json.loads(original.backend_tokenizer.to_str())
    keys = ("normalizer", "pre_tokenizer", "decoder", "post_processor", "added_tokens")
    tokenizers = {}
    for name, path in {"base": directory / "base_tokenizer.json", **langmap_paths(directory)}.items():
        tokenizer = Tokenizer.from_file(str(path))
        if tokenizer.get_vocab() != expected:
            raise ValueError(f"{path}: token-to-ID mapping differs from Original")
        saved = json.loads(tokenizer.to_str())
        for key in keys:
            if config.get(key) != saved.get(key):
                raise ValueError(f"{path}: changed tokenizer component {key}")
        tokenizers[name] = tokenizer
    return tokenizers


def fit(args):
    # The CLI exposes the pinned upstream checkout on PYTHONPATH for this command.
    from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer
    from code.standard_unigramlm import StandardUnigramLMTokenizer

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    corpus = args.run / "corpus"
    train = load_split(corpus, "train.jsonl")
    heldout = load_split(corpus, "heldout.jsonl")
    out = args.run / "langmap"
    out.mkdir(exist_ok=False)
    work = out / "corpora"
    work.mkdir()
    original = original_tokenizer()
    original.save_pretrained(out / "model_tokenizer")
    fit_counts = {}
    corpora = {}
    for lang in LANGUAGES:
        records = [row for row in train if row["lang"] == lang]
        selected = records[:args.fit_docs] if args.fit_docs else records
        fit_counts[lang] = len(selected)
        path = work / f"{lang}.txt"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in selected:
                for line in row["text"].split("\n"):
                    # Leave room for ByteLevel expansion under SentencePiece's line limit.
                    for start in range(0, len(line), 512):
                        segment = line[start:start + 512]
                        if segment.strip():
                            stream.write(segment + "\n")
        corpora[lang] = str(path)

    start = perf_counter()
    base_path = out / "base_tokenizer.json"
    seed_path = out / "model_tokenizer" / "tokenizer.json"
    unknown = original.unk_token or original.eos_token
    base = StandardUnigramLMTokenizer(from_base_tokenizer=str(seed_path), unk_token=unknown,
                                      em_mode="soft", num_iterations=args.em_iterations)
    base.train(list(corpora.values()), str(base_path), corpus_info=corpora)
    las = LanguageSpecificUnigramLMTokenizer(unk_token=unknown, reestimation_em_mode="soft",
                                            num_iterations=args.em_iterations)
    las.train(corpus_info=corpora, base_tokenizer_path=str(base_path), output_dir=str(out),
              tokenizer_path_format=str(out / "langspec_{lang_code}.json"),
              load_tokenizers_if_exists=False, max_workers=1)

    tokenizers = checked_langmap(out, original)
    for _ in encode_documents(heldout, original.backend_tokenizer):
        pass
    for _ in encode_documents(heldout, tokenizers):
        pass
    write_json(out / "manifest.json", {
        "unimixlm_revision": LANGMAP_REVISION,
        "model_revision": MODEL_REVISION,
        "train_sha256": sha256_file(corpus / "train.jsonl"),
        "em_iterations_requested": args.em_iterations,
        "fit_seconds": perf_counter() - start,
        "fit_documents_per_language": fit_counts,
        "tokenizer_sha256": {p.name: sha256_file(p) for p in [base_path, *langmap_paths(out).values()]},
    })


def verified_langmap_directory(run):
    directory = run / "langmap"
    manifest = read_json(directory / "manifest.json")
    if manifest["model_revision"] != MODEL_REVISION:
        raise ValueError("LangMAP model revision differs from the training model revision")
    if manifest["train_sha256"] != sha256_file(run / "corpus" / "train.jsonl"):
        raise ValueError("LangMAP was fitted on a different training corpus")
    for filename, digest in manifest["tokenizer_sha256"].items():
        if sha256_file(directory / filename) != digest:
            raise ValueError(f"LangMAP checksum mismatch: {filename}")
    return directory


def verified_tokenizers(run, original):
    return checked_langmap(verified_langmap_directory(run), original)


def encode_documents(documents, tokenizers):
    """Return language and token IDs, rejecting lossy encodings."""
    for row in documents:
        lang, text = row["lang"], row["text"]
        tokenizer = tokenizers[lang] if isinstance(tokenizers, dict) else tokenizers
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if not ids or tokenizer.decode(ids, skip_special_tokens=False) != text:
            raise ValueError(f"Round-trip failed for {lang} document {row['id']}; "
                             "check Unicode normalization and special-token handling")
        yield lang, ids
