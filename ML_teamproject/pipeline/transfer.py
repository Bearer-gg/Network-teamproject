from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from .versioning import normalize_version


def _ssh_target(host: str, username: str | None) -> str:
    return f"{username}@{host}" if username else host


def _ssh_options(identity_file: str | None, port: int | None) -> list[str]:
    options: list[str] = []
    if identity_file:
        options.extend(["-i", identity_file])
    if port:
        options.extend(["-P", str(port)])
    return options


def fetch_data(
    *,
    dataset_version: str,
    host: str,
    remote_capture_dir: str,
    local_raw_dir: str | Path,
    username: str | None = None,
    identity_file: str | None = None,
    port: int | None = None,
    overwrite: bool = False,
) -> Path:
    resolved_version = normalize_version(dataset_version)
    filename = f"dataset_{resolved_version}.pcap"
    destination = Path(local_raw_dir).resolve() / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Local pcap already exists: {destination}")
    remote_path = f"{remote_capture_dir.rstrip('/')}/{filename}"
    source = f"{_ssh_target(host, username)}:{remote_path}"
    command = ["scp", *_ssh_options(identity_file, port), source, str(destination)]
    subprocess.run(command, check=True)
    if not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError(f"Transfer did not produce a non-empty pcap: {destination}")
    return destination


def deploy_model(
    *,
    model_path: str | Path,
    host: str,
    remote_model_dir: str,
    username: str | None = None,
    identity_file: str | None = None,
    port: int | None = None,
    overwrite: bool = False,
) -> str:
    model = Path(model_path).resolve()
    if not model.exists():
        raise FileNotFoundError(f"Model does not exist: {model}")
    remote_path = f"{remote_model_dir.rstrip('/')}/{model.name}"
    target = _ssh_target(host, username)
    ssh_options: list[str] = []
    if identity_file:
        ssh_options.extend(["-i", identity_file])
    if port:
        ssh_options.extend(["-p", str(port)])
    check = subprocess.run(
        ["ssh", *ssh_options, target, f"test -e {shlex.quote(remote_path)}"],
        check=False,
    )
    if check.returncode == 0 and not overwrite:
        raise FileExistsError(f"Remote model already exists: {remote_path}")
    command = [
        "scp",
        *_ssh_options(identity_file, port),
        str(model),
        f"{target}:{remote_path}",
    ]
    subprocess.run(command, check=True)
    return remote_path
