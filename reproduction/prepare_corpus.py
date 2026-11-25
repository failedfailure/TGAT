"""Rebuild the paper's corpus from the archived upstream revisions."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from tgat_phase1.data import LANGUAGES, document_splits, read_json, sha256_file, write_splits


MANIFEST = Path(__file__).with_name("paper-corpus-manifest.json")
TRAIN_FILE = "train.jsonl"
HOLDOUT_FILE = "heldout.jsonl"


def verify(corpus, manifest):
    if read_json(corpus / "manifest.json") != manifest:
        raise ValueError(f"Corpus manifest differs from the paper: {corpus}")
    for filename, record in manifest["splits"].items():
        if sha256_file(corpus / filename) != record["sha256"]:
            raise ValueError(f"Corpus checksum differs from the paper: {corpus / filename}")


def prepare(run):
    manifest = read_json(MANIFEST)
    corpus = run / "corpus"
    if corpus.exists():
        verify(corpus, manifest)
        print(f"Verified existing paper corpus: {corpus}")
        return

    from datasets import load_dataset

    sources = manifest["sources"]
    streams = {
        lang: load_dataset(
            sources[lang]["repository"], sources[lang]["config"],
            revision=sources[lang]["revision"], split="train", streaming=True,
        )
        for lang in LANGUAGES
    }
    train_docs = manifest["splits"][TRAIN_FILE]["documents"] // len(LANGUAGES)
    holdout_docs = manifest["splits"][HOLDOUT_FILE]["documents"] // len(LANGUAGES)
    seed = manifest["split_seed"]
    train, heldout = document_splits(streams, train_docs, holdout_docs, seed)

    run.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".paper-corpus-", dir=run) as directory:
        staged = Path(directory) / "corpus"
        write_splits(staged, {"sources": sources, "split_seed": seed},
                     {TRAIN_FILE: train, HOLDOUT_FILE: heldout})
        verify(staged, manifest)
        staged.rename(corpus)
    print(f"Rebuilt and verified paper corpus: {corpus}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/pilot"))
    prepare(parser.parse_args().run)


if __name__ == "__main__":
    main()
