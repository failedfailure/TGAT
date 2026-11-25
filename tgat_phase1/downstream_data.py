"""Freeze the multilingual grammaticality and English BLiMP retention panels."""

import csv
from collections import Counter
from hashlib import sha1, sha256
from io import BytesIO, StringIO
import json
from pathlib import PurePosixPath
from tempfile import TemporaryDirectory
from unicodedata import normalize
from urllib.request import urlopen
from zipfile import ZipFile


MULTIBLIMP_REVISION = "de923efa8d2483d6b13364ee68e65308e990a991"
JBLIMP_REVISION = "87ea6d2c1b67e5305789f4815f3a06bd1ec37aa2"
JBLIMP_GIT_BLOB = "200aab6b1a013fb98def4c7bfcfc978b26a7797a"
BLIMP_REVISION = "3e56b06fcabca9b30822fc66435fca6b1aa40bb1"
BLIMP_PER_PARADIGM = 20
BLIMP_PARADIGMS = 67
PAIR_FIELDS = {"dataset", "lang", "phenomenon", "id", "good", "bad"}
EXPECTED_COUNTS = {
    "multiblimp/eng_Latn": 770, "multiblimp/tur_Latn": 1742, "jblimp/jpn_Jpan": 331,
    "blimp/eng_Latn": BLIMP_PARADIGMS * BLIMP_PER_PARADIGM,
}
REVISIONS = {"multiblimp": MULTIBLIMP_REVISION, "jblimp": JBLIMP_REVISION, "blimp": BLIMP_REVISION}
LICENSES = {"multiblimp": "CC-BY-4.0", "jblimp": "CC-BY-SA-4.0", "blimp": "CC-BY-4.0"}
SOURCES = {
    "multiblimp/eng_Latn": f"https://huggingface.co/datasets/jumelet/multiblimp/resolve/{MULTIBLIMP_REVISION}/eng/data.tsv",
    "multiblimp/tur_Latn": f"https://huggingface.co/datasets/jumelet/multiblimp/resolve/{MULTIBLIMP_REVISION}/tur/data.tsv",
    "jblimp/jpn_Jpan": f"https://raw.githubusercontent.com/osekilab/JBLiMP/{JBLIMP_REVISION}/data/validated/validated_minimal_pairs.jsonl",
    "blimp/eng_Latn": f"https://codeload.github.com/alexwarstadt/blimp/zip/{BLIMP_REVISION}",
}


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected a nonempty text field in the benchmark")
    return normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))


def _pair(dataset, lang, phenomenon, pair_id, good, bad):
    return {"dataset": dataset, "lang": lang, "phenomenon": _text(phenomenon),
            "id": _text(pair_id), "good": _text(good), "bad": _text(bad)}


def _jsonl(data):
    for line in StringIO(data.decode("utf-8-sig")):
        if line.strip():
            yield json.loads(line)


def _multiblimp(data, lang):
    source_lang = lang.split("_", 1)[0]
    rows = csv.DictReader(StringIO(data.decode("utf-8-sig"), newline=""), delimiter="\t", strict=True)
    if not {"sen", "wrong_sen", "phenomenon", "lang"} <= set(rows.fieldnames or ()):
        raise ValueError(f"MultiBLiMP {lang}: missing source columns")
    for i, row in enumerate(rows, start=1):
        if row["lang"] != source_lang or None in row:
            raise ValueError(f"MultiBLiMP {lang}: unexpected source language or malformed TSV row {i}")
        yield _pair("multiblimp", lang, row["phenomenon"], f"{source_lang}:{i:04d}",
                    row["sen"], row["wrong_sen"])


def _jblimp(data):
    for row in _jsonl(data):
        if type(row["ID"]) is not int:
            raise ValueError("JBLiMP: expected an integer source ID")
        yield _pair("jblimp", "jpn_Jpan", row["phenomenon"], str(row["ID"]),
                    row["good_sentence"], row["bad_sentence"])


def _blimp_paradigm(data, uid, size=BLIMP_PER_PARADIGM):
    ranked = []
    ids = set()
    for row in _jsonl(data):
        pair_id = row["pairID"]
        if row["UID"] != uid or not isinstance(pair_id, str) or not pair_id.isdecimal() or pair_id in ids:
            raise ValueError(f"BLiMP {uid}: unexpected UID, invalid or duplicate pairID")
        ids.add(pair_id)
        key = json.dumps((uid, pair_id), ensure_ascii=True, separators=(",", ":")).encode("ascii")
        ranked.append(
            (sha256(key).digest(), pair_id, _pair("blimp", "eng_Latn", uid, f"{uid}:{pair_id}",
                                                 row["sentence_good"], row["sentence_bad"]))
        )
    if len(ranked) != 1000:
        raise ValueError(f"BLiMP {uid}: expected 1,000 source pairs, found {len(ranked)}")
    return [row for _, _, row in sorted(ranked)[:size]]


def _blimp(data):
    rows = []
    file_hashes = {}
    with ZipFile(BytesIO(data)) as archive:
        for name in sorted(archive.namelist()):
            path = PurePosixPath(name)
            if path.parts[:2] != (f"blimp-{BLIMP_REVISION}", "data") or len(path.parts) != 3 or path.suffix != ".jsonl":
                continue
            content = archive.read(name)
            rows.extend(_blimp_paradigm(content, path.stem))
            file_hashes[path.name] = sha256(content).hexdigest()
    if len(file_hashes) != BLIMP_PARADIGMS:
        raise ValueError(f"BLiMP: expected {BLIMP_PARADIGMS} paradigms, found {len(file_hashes)}")
    return rows, file_hashes


def _check_panel(rows):
    ids = set()
    counts = Counter()
    paradigms = Counter()
    for row in rows:
        if set(row) != PAIR_FIELDS or any(not isinstance(value, str) or _text(value) != value for value in row.values()):
            raise ValueError("Invalid normalized downstream pair")
        dataset, lang, pair_id = (row[key] for key in ("dataset", "lang", "id"))
        key = dataset, lang, pair_id
        if key in ids:
            raise ValueError(f"Duplicate downstream pair {key}")
        ids.add(key)
        counts[f"{dataset}/{lang}"] += 1
        if dataset == "blimp":
            paradigms[row["phenomenon"]] += 1
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected downstream panel counts: {dict(counts)}")
    if len(paradigms) != BLIMP_PARADIGMS or set(paradigms.values()) != {BLIMP_PER_PARADIGM}:
        raise ValueError("English BLiMP panel does not contain 20 pairs per paradigm")
    return dict(counts)


def _download(url):
    with urlopen(url, timeout=60) as stream:
        return stream.read()


def prepare(output):
    """Download pinned public source files and atomically publish a frozen panel."""
    if output.exists():
        load_pairs(output)
        return

    raw = {key: _download(url) for key, url in SOURCES.items()}
    jblimp = raw["jblimp/jpn_Jpan"]
    git_blob = sha1(f"blob {len(jblimp)}\0".encode("ascii") + jblimp).hexdigest()
    if git_blob != JBLIMP_GIT_BLOB:
        raise ValueError("JBLiMP validated JSONL differs from the pinned Git blob")
    blimp_rows, blimp_files = _blimp(raw["blimp/eng_Latn"])
    rows = [
        *_multiblimp(raw["multiblimp/eng_Latn"], "eng_Latn"),
        *_multiblimp(raw["multiblimp/tur_Latn"], "tur_Latn"),
        *_jblimp(jblimp),
        *blimp_rows,
    ]
    counts = _check_panel(rows)
    rows.sort(key=lambda row: (row["dataset"], row["lang"], row["phenomenon"], row["id"]))
    content = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows).encode("utf-8")
    manifest = {
        "schema_version": 1, "revision": REVISIONS, "licenses": LICENSES, "counts": counts,
        "sources": {
            key: {"url": url, "bytes": len(raw[key]), "sha256": sha256(raw[key]).hexdigest()}
            for key, url in SOURCES.items()
        },
        "blimp_file_sha256": blimp_files,
        "blimp_sampling": "20 ascending SHA-256 digests per UID of compact JSON [UID,pairID]",
        "text_normalization": "NFC and LF line endings",
        "pairs_sha256": sha256(content).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".downstream-data-", dir=output.parent) as temporary:
        staged = output.parent / PurePosixPath(temporary).name / "panel"
        staged.mkdir()
        (staged / "pairs.jsonl").write_bytes(content)
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        staged.rename(output)


def load_pairs(output):
    """Read the frozen panel after verifying its revision, content hash, and shape."""
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("revision") != REVISIONS
            or manifest.get("licenses") != LICENSES or manifest.get("counts") != EXPECTED_COUNTS):
        raise ValueError("Downstream manifest differs from the pinned panel definition")
    content = (output / "pairs.jsonl").read_bytes()
    if sha256(content).hexdigest() != manifest["pairs_sha256"]:
        raise ValueError("Frozen downstream pairs failed SHA-256 verification")
    rows = list(_jsonl(content))
    _check_panel(rows)
    return rows
