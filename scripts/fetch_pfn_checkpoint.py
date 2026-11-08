"""Explicitly cache the pinned official HEBO+ release, verifying before unpickling."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen

from models.pfn_backend import (CHECKPOINT_FILENAME, CHECKPOINT_SHA256, COMPRESSED_BLOB_SHA1,
    COMPRESSED_SHA256, DEFAULT_CHECKPOINT, SOURCE_URI, checkpoint_metadata)


def main():
    model = DEFAULT_CHECKPOINT
    model.parent.mkdir(parents=True, exist_ok=True)
    compressed = model.with_suffix(".pt.gz")
    if not model.exists():
        if not compressed.exists():
            temporary = compressed.with_suffix(".gz.partial")
            with urlopen(SOURCE_URI, timeout=120) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
            temporary.replace(compressed)
        raw = compressed.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if (len(raw) != 99440314 or hashlib.sha256(raw).hexdigest() != COMPRESSED_SHA256
                or blob != COMPRESSED_BLOB_SHA1):
            raise ValueError("Official checkpoint gzip integrity check failed; no pickle will be loaded")
        temporary = model.with_suffix(".pt.partial")
        with gzip.open(compressed, "rb") as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
            raise ValueError("Uncompressed official model checksum mismatch")
        temporary.replace(model)
    info = checkpoint_metadata()
    (model.parent / "hebo_plus.provenance.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
