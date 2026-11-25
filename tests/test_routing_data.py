from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tgat_phase1 import routing_data as data


class RoutingDataTests(unittest.TestCase):
    def test_frozen_rows_keep_line_alignment_and_article_attribution(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "source.tar.gz"
            output = root / "frozen"
            with tarfile.open(archive, "w:gz") as bundle:
                for name, (split, lang) in data.MEMBERS.items():
                    self.add_member(bundle, name, f" {lang}:{split} e\u0301 \r\n".encode())
                header = "URL\tdomain\ttopic\thas_image\thas_hyperlink\n"
                for name, split in data.METADATA.items():
                    record = f"https://example.org/{split}\twikinews\tculture\t0\t1\n"
                    self.add_member(bundle, name, (header + record).encode())
            with patch.object(data, "ARCHIVE_SHA256", sha256(archive.read_bytes()).hexdigest()), \
                    patch.dict(data.SPLIT_COUNTS, {"dev": 1, "devtest": 1}, clear=True):
                data.prepare(output, archive)
                data.prepare(output, archive)
                rows = data.load_rows(output)
                self.assertEqual(len(rows), 6)
                self.assertEqual(rows[1], {"id": "dev:0001", "split": "dev", "lang": "jpn_Jpan",
                                           "text": " jpn_Jpan:dev é "})
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertEqual(manifest["source_articles"]["devtest:0001"], {
                    "url": "https://example.org/devtest", "domain": "wikinews", "topic": "culture",
                })
                frozen = output / "rows.jsonl"
                frozen.write_bytes(frozen.read_bytes() + b"\n")
                with self.assertRaisesRegex(ValueError, "failed SHA-256"):
                    data.load_rows(output)

    def test_supplied_unpinned_archive_does_not_publish_panel(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "source.tar.gz"
            archive.write_bytes(b"replacement archive")
            output = root / "frozen"
            with self.assertRaisesRegex(ValueError, "pinned SHA-256"):
                data.prepare(output, archive)
            self.assertFalse(output.exists())

    @staticmethod
    def add_member(bundle, name, payload):
        member = tarfile.TarInfo("./" + name)
        member.size = len(payload)
        bundle.addfile(member, BytesIO(payload))


if __name__ == "__main__":
    unittest.main()
