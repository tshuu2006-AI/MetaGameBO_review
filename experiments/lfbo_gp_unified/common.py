"""Stdlib file, checksum and path helpers for LFBO/GP."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def safe_id(value):
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', value) or value in ('.', '..'):
        raise ValueError('run-id chỉ gồm chữ/số/._-, bắt đầu bằng chữ/số, tối đa 80 ký tự')
    return value

def safe_child(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe artifact path')
    path = Path(root) / relative
    require(path.resolve().is_relative_to(Path(root).resolve()), 'Artifact escapes output folder')
    return path

def write_text(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        return
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def write_csv(path, records, fields, *, replace=False):
    import io
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
    write_text(path, stream.getvalue(), replace=replace)
