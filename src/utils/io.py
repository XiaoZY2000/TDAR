# src/utils/io.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union, Optional

import torch


PathLike = Union[str, Path]


# =========================
# Path utilities
# =========================

def ensure_dir(p: PathLike):
    """
    Ensure directory exists. Create parents if needed.
    NOTE: As per project convention, call this ONLY for subdirectories,
    not for top-level dirs like data/ or checkpoints/.
    """
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)


def ensure_parent_dir(file_path: PathLike):
    """
    Ensure parent directory of a file path exists.
    """
    p = Path(file_path)
    if p.parent is not None:
        p.parent.mkdir(parents=True, exist_ok=True)


# =========================
# JSON
# =========================

def load_json(path: PathLike) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: PathLike, indent: int = 2, ensure_ascii: bool = False):
    path = Path(path)
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=ensure_ascii, indent=indent)


# =========================
# JSON Lines (JSONL)
# =========================

def load_jsonl(path: PathLike) -> List[Any]:
    path = Path(path)
    data: List[Any] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def iter_jsonl(path: PathLike):
    """
    Generator version to stream large jsonl files.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def save_jsonl(objs: Iterable[Any], path: PathLike):
    path = Path(path)
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")


# =========================
# TSV / CSV-like (tab-separated)
# =========================

def load_tsv(path: PathLike, sep: str = "\t") -> List[List[str]]:
    path = Path(path)
    rows: List[List[str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            rows.append(line.split(sep))
    return rows


def save_tsv(rows: Iterable[Iterable[Any]], path: PathLike, sep: str = "\t"):
    path = Path(path)
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            line = sep.join(str(x) for x in row)
            f.write(line)
            f.write("\n")


# =========================
# Plain text lines
# =========================

def load_lines(path: PathLike) -> List[str]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def save_lines(lines: Iterable[str], path: PathLike):
    path = Path(path)
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line))
            f.write("\n")


# =========================
# Torch checkpoint helpers
# =========================

def torch_save(obj: Any, path: PathLike):
    """
    Wrapper around torch.save with parent dir creation.
    """
    path = Path(path)
    ensure_parent_dir(path)
    torch.save(obj, path)


def torch_load(path: PathLike, map_location: Optional[str] = "cpu") -> Any:
    path = Path(path)
    if map_location is None:
        return torch.load(path)
    return torch.load(path, map_location=map_location)


# =========================
# Convenience: pretty print dict to json
# =========================

def dump_config(cfg: Dict[str, Any], path: PathLike):
    """
    Save config dict nicely as json.
    """
    save_json(cfg, path, indent=2, ensure_ascii=False)
