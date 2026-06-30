from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml
from scapy.all import DNS, IP, IPv6, TCP, UDP, PcapReader, bind_layers

from .features import extract_feature_row
from .schema import CSV_COLUMNS
from .versioning import resolve_version


MONITORED_PORTS = (10053, 20053, 30053, 1025)
LABEL_DOMAIN_MARKER = "bank.test"
LABEL_ATTACK_IP = "192.168.219.104"
CAPTURE_SCOPES = ("resolver-bound", "all-dns-responses")


def bind_dns_ports() -> None:
    for port in MONITORED_PORTS:
        bind_layers(UDP, DNS, sport=port)
        bind_layers(UDP, DNS, dport=port)
        bind_layers(TCP, DNS, sport=port)
        bind_layers(TCP, DNS, dport=port)


def is_in_capture_scope(packet, resolver_ip: str | None, capture_scope: str = "resolver-bound") -> bool:
    if not packet.haslayer(DNS) or int(packet[DNS].qr or 0) != 1:
        return False
    if capture_scope == "all-dns-responses":
        return True
    if capture_scope != "resolver-bound":
        raise ValueError(f"Unknown capture scope: {capture_scope}")
    if not resolver_ip:
        raise ValueError("resolver_ip is required for resolver-bound capture scope.")
    ip = packet[IP] if packet.haslayer(IP) else None
    ipv6 = packet[IPv6] if packet.haslayer(IPv6) else None
    dst_ip = str(ip.dst if ip else ipv6.dst if ipv6 else "")
    if dst_ip != resolver_ip:
        return False
    udp = packet[UDP] if packet.haslayer(UDP) else None
    tcp = packet[TCP] if packet.haslayer(TCP) else None
    src_port = int(udp.sport if udp else tcp.sport if tcp else 0)
    dst_port = int(udp.dport if udp else tcp.dport if tcp else 0)
    return src_port in MONITORED_PORTS or dst_port in MONITORED_PORTS


def output_paths(project_root: Path, version: str) -> tuple[Path, Path]:
    csv_path = project_root / "data" / "processed" / f"features_{version}.csv"
    manifest_path = project_root / "data" / "manifests" / f"dataset_{version}.yaml"
    return csv_path, manifest_path


def build_dataset(
    *,
    pcap_path: str | Path,
    resolver_ip: str | None,
    capture_scope: str = "resolver-bound",
    version: str | None = None,
    project_root: str | Path = ".",
    overwrite: bool = False,
) -> dict:
    root = Path(project_root).resolve()
    source = Path(pcap_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"PCAP does not exist: {source}")
    resolved_version = resolve_version(source, version)
    if capture_scope not in CAPTURE_SCOPES:
        raise ValueError(f"Unknown capture scope {capture_scope!r}; expected one of {CAPTURE_SCOPES}.")
    if capture_scope == "resolver-bound" and not resolver_ip:
        raise ValueError("resolver_ip is required for resolver-bound capture scope.")
    csv_path, manifest_path = output_paths(root, resolved_version)
    if not overwrite:
        for path in (csv_path, manifest_path):
            if path.exists():
                raise FileExistsError(f"Output already exists: {path}; pass overwrite=True to replace it.")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    bind_dns_ports()
    counts: Counter[int] = Counter()
    total_packets = 0
    accepted_rows = 0

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        with PcapReader(str(source)) as reader:
            for packet_index, packet in enumerate(reader):
                total_packets += 1
                if not is_in_capture_scope(packet, resolver_ip, capture_scope):
                    continue
                row = extract_feature_row(
                    packet,
                    packet_index,
                    source_pcap_version=resolved_version,
                    source_pcap_file=source.name,
                    domain_marker=LABEL_DOMAIN_MARKER,
                    attack_ip=LABEL_ATTACK_IP,
                )
                writer.writerow(row)
                accepted_rows += 1
                counts[int(row["label"])] += 1

    if accepted_rows == 0:
        csv_path.unlink(missing_ok=True)
        raise ValueError(
            "No packets matched the training scope: "
            + (
                "DNS responses."
                if capture_scope == "all-dns-responses"
                else f"DNS response, destination resolver IP, and monitored ports {MONITORED_PORTS}."
            )
        )

    manifest = {
        "dataset_version": resolved_version,
        "source_pcap_files": [str(source)],
        "processed_csv": str(csv_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "capture_scope": {
            "mode": capture_scope,
            "resolver_ip": resolver_ip,
            "dns_response_only": True,
            "monitored_ports": list(MONITORED_PORTS) if capture_scope == "resolver-bound" else None,
        },
        "label_mapping": {
            0: "normal",
            1: "attack",
            "rule": (
                "label=1 only when any DNS question/RR name/value contains "
                f"{LABEL_DOMAIN_MARKER!r} and any DNS RR value contains {LABEL_ATTACK_IP!r}"
            ),
        },
        "row_counts": {
            "packets_read": total_packets,
            "accepted_rows": accepted_rows,
            "normal": counts.get(0, 0),
            "attack": counts.get(1, 0),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False, allow_unicode=False)

    return {
        "dataset_version": resolved_version,
        "csv_path": csv_path,
        "manifest_path": manifest_path,
        "total_packets": total_packets,
        "accepted_rows": accepted_rows,
        "label_counts": dict(sorted(counts.items())),
    }
