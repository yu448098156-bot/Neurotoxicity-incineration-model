"""Portable paths, CSV/JSON output and experiment provenance."""
from __future__ import annotations
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def load_config(path: str | Path | None = None) -> dict:
    source = Path(path).resolve() if path else ROOT / 'config/models.json'
    return json.loads(source.read_text(encoding='utf-8'))

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f'Cannot serialise {type(value)}')
    def clean(v):
        if isinstance(v, dict): return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v, (list,tuple)): return [clean(x) for x in v]
        if isinstance(v, np.ndarray): return clean(v.tolist())
        if isinstance(v, np.generic): return clean(v.item())
        if isinstance(v, float) and not np.isfinite(v): return str(v)
        return v
    path.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False, default=convert,
                               allow_nan=False) + '\n', encoding='utf-8')

def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)

def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f'Empty CSV: {path}')
        return list(reader.fieldnames), list(reader)

def environment() -> dict:
    packages = {}
    for package in ['numpy','scipy','pandas','scikit-learn','rdkit','catboost',
                    'xgboost','shap','joblib','matplotlib','threadpoolctl']:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {'python':sys.version, 'platform':platform.platform(),
            'machine':platform.machine(), 'packages':packages,
            'cpu_count_reported':os.cpu_count(),
            'timestamp_utc':datetime.now(timezone.utc).isoformat()}

def prepare_output(path: Path, overwrite: bool = False) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f'Output directory is not empty: {path}. Use a new directory or --overwrite.')
    # Do not delete arbitrary directories; callers overwrite only named files.
    path.mkdir(parents=True, exist_ok=True)
