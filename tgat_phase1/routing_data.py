"""Freeze aligned FLORES-200 dev and devtest sentences for routing evaluation."""

import csv
from hashlib import file_digest, sha256
from io import StringIO
import json
from pathlib import Path, PurePosixPath
from shutil import copyfileobj
import tarfile
from tempfile import TemporaryDirectory
from unicodedata import normalize
from urllib.request import urlopen


LANGUAGES = ("eng_Latn", "jpn_Jpan", "tur_Latn")
SPLIT_COUNTS = {"dev": 997, "devtest": 1012}
ARCHIVE_SHA256 = "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"
SOURCE = {
    "dataset": "FLORES-200",
    "version": "Meta original public release",
    "license": "CC-BY-SA-4.0",
    "url": "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz",
    "documentation": "https://github.com/facebookresearch/flores/blob/a6c830c6e1051fb4ac1a44b32358f00463f332bd/flores200/README.md",
}
ROW_FIELDS = {"id", "split", "lang", "text"}
MEMBERS = {
    f"flores200_dataset/{split}/{lang}.{split}": (split, lang)
    for split in SPLIT_COUNTS for lang in LANGUAGES
}
METADATA = {f"flores200_dataset/metadata_{split}.tsv": split for split in SPLIT_COUNTS}
PUBLISHER_FILES = MEMBERS.keys() | METADATA.keys()


def _sha256(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _file_payloads(archive):
    payloads = {}
    # Read the six language files and their two metadata files without extracting.
    with tarfile.open(archive, mode="r|gz") as bundle:
        for member in bundle:
            name = str(PurePosixPath(member.name))
            if name not in PUBLISHER_FILES:
                continue
            if not member.isfile() or name in payloads:
                raise ValueError(f"Unexpected or repeated FLORES-200 archive member {name}")
            with bundle.extractfile(member) as stream:
                payloads[name] = stream.read()
    if missing := PUBLISHER_FILES - payloads.keys():
        raise ValueError(f"Missing FLORES-200 archive members: {', '.join(sorted(missing))}")
    return payloads


def _sentences(payload, split, lang):
    decoded = payload.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    sentences = [normalize("NFC", line) for line in decoded.removesuffix("\n").split("\n")]
    if len(sentences) != SPLIT_COUNTS[split] or any(not sentence.strip() for sentence in sentences):
        raise ValueError(f"FLORES-200 {split} {lang}: expected {SPLIT_COUNTS[split]} nonempty sentences")
    return sentences


def _coordinates():
    for split, count in SPLIT_COUNTS.items():
        for index in range(count):
            for lang in LANGUAGES:
                yield _row_id(split, index), split, index, lang


def _row_id(split, index):
    return f"{split}:{index + 1:04d}"


def _aligned_rows(payloads):
    parallel = {
        (split, lang): _sentences(payloads[name], split, lang)
        for name, (split, lang) in MEMBERS.items()
    }
    return [
        {"id": row_id, "split": split, "lang": lang, "text": parallel[split, lang][index]}
        for row_id, split, index, lang in _coordinates()
    ]


def _articles(payloads):
    articles = {}
    for name, split in METADATA.items():
        text = payloads[name].decode("utf-8-sig")
        records = csv.DictReader(StringIO(text, newline=""), delimiter="\t", strict=True)
        if not {"URL", "domain", "topic"} <= set(records.fieldnames or ()):
            raise ValueError(f"FLORES-200 {split}: missing article metadata columns")
        rows = list(records)
        if len(rows) != SPLIT_COUNTS[split]:
            raise ValueError(f"FLORES-200 {split}: article metadata and sentence counts differ")
        for index, row in enumerate(rows):
            if None in row or not (row["URL"] or "").startswith(("https://", "http://")):
                raise ValueError(f"FLORES-200 {split}: invalid article metadata for row {index + 1}")
            articles[_row_id(split, index)] = {"url": row["URL"], "domain": row["domain"], "topic": row["topic"]}
    return articles


def prepare(output: Path, archive: Path | None = None):
    """Freeze six files from the official archive, including aligned row IDs."""
    if output.exists():
        load_rows(output)
        if archive is not None:
            existing = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            if existing["archive_sha256"] != _sha256(archive):
                raise ValueError("Existing panel was built from a different FLORES-200 archive")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".flores200-", dir=output.parent) as temporary:
        work = Path(temporary)
        if archive is None:
            archive = work / "flores200_dataset.tar.gz"
            with urlopen(SOURCE["url"], timeout=120) as response, archive.open("wb") as downloaded:
                copyfileobj(response, downloaded)
        archive_digest = _sha256(archive)
        if archive_digest != ARCHIVE_SHA256:
            raise ValueError("FLORES-200 archive does not match the pinned SHA-256")
        payloads = _file_payloads(archive)
        rows = _aligned_rows(payloads)
        staged = work / "panel"
        staged.mkdir()
        frozen = staged / "rows.jsonl"
        with frozen.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        manifest = {
            "schema_version": 1, "source": SOURCE, "archive_sha256": archive_digest,
            "archive_bytes": archive.stat().st_size, "languages": list(LANGUAGES),
            "split_counts": SPLIT_COUNTS, "rows": len(rows), "rows_sha256": _sha256(frozen),
            "members_sha256": {name: sha256(content).hexdigest() for name, content in payloads.items()},
            "source_articles": _articles(payloads),
            "alignment": "1-based line number within each split",
            "text_normalization": "UTF-8, NFC and LF line endings, preserve sentence spacing",
        }
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        staged.rename(output)


def load_rows(output: Path):
    """Read frozen rows and verify strict split/row/language alignment."""
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("source") != SOURCE
            or manifest.get("archive_sha256") != ARCHIVE_SHA256
            or manifest.get("languages") != list(LANGUAGES) or manifest.get("split_counts") != SPLIT_COUNTS
            or manifest.get("members_sha256", {}).keys() != PUBLISHER_FILES):
        raise ValueError("Routing data manifest differs from the pinned FLORES-200 definition")
    frozen = output / "rows.jsonl"
    if _sha256(frozen) != manifest["rows_sha256"]:
        raise ValueError("Frozen FLORES-200 rows failed SHA-256 verification")
    with frozen.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != manifest.get("rows") or len(rows) != sum(SPLIT_COUNTS.values()) * len(LANGUAGES):
        raise ValueError("Invalid FLORES-200 row count")
    if manifest.get("source_articles", {}).keys() != {row_id for row_id, _, _, _ in _coordinates()}:
        raise ValueError("FLORES-200 article metadata does not cover the aligned sentence IDs")
    for row, (row_id, split, _, lang) in zip(rows, _coordinates(), strict=True):
        if (row.keys() != ROW_FIELDS or (row["id"], row["split"], row["lang"]) != (row_id, split, lang)
                or not isinstance(row["text"], str) or not row["text"].strip()
                or row["text"] != normalize("NFC", row["text"]) or "\r" in row["text"] or "\n" in row["text"]):
            raise ValueError(f"Invalid frozen FLORES-200 row {row_id}/{lang}")
    return rows
