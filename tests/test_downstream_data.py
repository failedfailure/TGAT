import csv
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tgat_phase1 import downstream_data as data


class DownstreamDataTests(unittest.TestCase):
    def test_multiblimp_handles_quoting_and_normalizes_without_stripping(self):
        tsv = StringIO(newline="")
        writer = csv.writer(tsv, delimiter="\t")
        writer.writerow(("lang", "sen", "wrong_sen", "phenomenon"))
        writer.writerow(("tur", ' e\u0301 "a"\r\nb ', "Bad", "SV-#"))
        raw = tsv.getvalue().encode("utf-8")
        self.assertEqual(list(data._multiblimp(raw, "tur_Latn")), [
            {"dataset": "multiblimp", "lang": "tur_Latn", "phenomenon": "SV-#", "id": "tur:0001",
             "good": ' é "a"\nb ', "bad": "Bad"},
        ])
        with self.assertRaisesRegex(ValueError, "unexpected source language"):
            list(data._multiblimp(raw, "eng_Latn"))

    def test_blimp_selects_lowest_id_tuple_hashes_in_any_input_order(self):
        uid = "subject_verb_agreement"
        lines = [json.dumps({"UID": uid, "pairID": str(i), "sentence_good": f"Good {i}",
                             "sentence_bad": f"Bad {i}"}) for i in range(1000)]
        selected = data._blimp_paradigm("\n".join(lines).encode(), uid)
        self.assertEqual(selected, data._blimp_paradigm("\n".join(reversed(lines)).encode(), uid))
        expected = sorted(
            range(1000), key=lambda i: sha256(f'["{uid}","{i}"]'.encode("ascii")).digest()
        )[:20]
        self.assertEqual([row["id"] for row in selected], [f"{uid}:{i}" for i in expected])

    def test_frozen_panel_rejects_changed_bytes(self):
        row = {"dataset": "blimp", "lang": "eng_Latn", "phenomenon": "uid", "id": "uid:0",
               "good": "Good", "bad": "Bad"}
        counts = {"blimp/eng_Latn": 1}
        payload = (json.dumps(row) + "\n").encode("utf-8")
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            pairs = output / "pairs.jsonl"
            pairs.write_bytes(payload)
            (output / "manifest.json").write_text(json.dumps({
                "schema_version": 1, "revision": data.REVISIONS, "licenses": data.LICENSES, "counts": counts,
                "pairs_sha256": sha256(payload).hexdigest(),
            }), encoding="utf-8")
            with patch.multiple(data, EXPECTED_COUNTS=counts, BLIMP_PARADIGMS=1, BLIMP_PER_PARADIGM=1):
                self.assertEqual(data.load_pairs(output), [row])
                pairs.write_bytes(payload + b"\n")
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    data.load_pairs(output)


if __name__ == "__main__":
    unittest.main()
