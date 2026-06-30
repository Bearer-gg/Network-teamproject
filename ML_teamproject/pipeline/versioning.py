from __future__ import annotations

import re
from pathlib import Path


VERSION_PATTERN = re.compile(r"^v?(\d{1,})$")
DATASET_PATTERN = re.compile(r"^dataset_(v\d{6})\.pcap$")
FEATURES_PATTERN = re.compile(r"^features_(v\d{6})\.csv$")
RUN_DIR_PATTERN = re.compile(r"^run_(v\d{6})$")


def normalize_version(value: str) -> str:
    match = VERSION_PATTERN.fullmatch(str(value).strip())
    if not match:
        raise ValueError(f"Invalid version {value!r}; expected vNNNNNN or a number.")
    number = int(match.group(1))
    return f"v{number:06d}"


def version_from_pcap(path: str | Path) -> str:
    match = DATASET_PATTERN.fullmatch(Path(path).name)
    if not match:
        raise ValueError(
            f"Cannot infer version from {Path(path).name!r}; "
            "use a dataset_vNNNNNN.pcap filename or pass --version."
        )
    return match.group(1)


def resolve_version(path: str | Path, version: str | None) -> str:
    return normalize_version(version) if version else version_from_pcap(path)


def dataset_version_from_features(path: str | Path) -> str:
    match = FEATURES_PATTERN.fullmatch(Path(path).name)
    if not match:
        raise ValueError(
            f"Cannot infer dataset version from {Path(path).name!r}; use features_vNNNNNN.csv."
        )
    return match.group(1)


def normalize_run_version(value: str) -> str:
    return normalize_version(value)


def next_run_version(root: str | Path) -> str:
    runs_dir = Path(root).resolve() / "runs"
    existing_numbers: list[int] = []
    if runs_dir.exists():
        for path in runs_dir.iterdir():
            match = RUN_DIR_PATTERN.fullmatch(path.name)
            if path.is_dir() and match:
                existing_numbers.append(int(match.group(1)[1:]))
    return normalize_run_version(str(max(existing_numbers, default=0) + 1))


def run_bundle_dir(root: str | Path, run_version: str) -> Path:
    return Path(root).resolve() / "runs" / f"run_{normalize_run_version(run_version)}"


def run_version_from_dir(path: str | Path) -> str:
    match = RUN_DIR_PATTERN.fullmatch(Path(path).name)
    if not match:
        raise ValueError(
            f"Cannot infer run version from {Path(path).name!r}; use run_vNNNNNN."
        )
    return match.group(1)
