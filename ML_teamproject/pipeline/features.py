from __future__ import annotations

import math
import time
from collections import Counter

from scapy.all import DNS, DNSRR, IP, IPv6, TCP, UDP

from .schema import CSV_COLUMNS


def normalize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value).strip().lower().rstrip(".")


def to_float(value, default=-1):
    try:
        return default if value is None else float(value)
    except Exception:
        return default


def to_int(value, default=-1):
    try:
        return default if value is None else int(str(value), 0)
    except Exception:
        return default


def entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    total = len(value)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def same_or_subdomain(child, parent) -> int:
    child = normalize_text(child)
    parent = normalize_text(parent)
    if not child or not parent:
        return 0
    return int(child == parent or child.endswith("." + parent))


def iter_rr(rrset, count):
    if isinstance(rrset, (list, tuple)):
        for rr in rrset[: int(count or 0)]:
            yield rr
        return
    current = rrset
    for _ in range(int(count or 0)):
        if current is None:
            return
        yield current
        current = getattr(current, "payload", None)
        if not isinstance(current, DNSRR):
            return


def first_question_name(dns) -> str:
    question = dns.getfieldval("qd")
    if question is None or int(getattr(dns, "qdcount", 0) or 0) <= 0:
        return ""
    if isinstance(question, (list, tuple)):
        question = question[0] if question else None
        if question is None:
            return ""
    try:
        return normalize_text(getattr(question, "qname", ""))
    except (IndexError, AttributeError, TypeError):
        return ""


def dns_type(rr) -> str:
    value = getattr(rr, "type", "")
    return str(int(value)) if isinstance(value, int) else str(value)


def collect_dns_records(dns) -> list[dict]:
    records = []
    for section, rrset, count in (
        ("answer", dns.getfieldval("an"), dns.ancount),
        ("authority", dns.getfieldval("ns"), dns.nscount),
        ("additional", dns.getfieldval("ar"), dns.arcount),
    ):
        for rr in iter_rr(rrset, count):
            records.append(
                {
                    "section": section,
                    "name": normalize_text(getattr(rr, "rrname", "")),
                    "type": dns_type(rr),
                    "ttl": to_float(getattr(rr, "ttl", None), default=math.nan),
                    "value": normalize_text(getattr(rr, "rdata", "")),
                }
            )
    return records


def apply_label_rule(dns, domain_marker: str, attack_ip: str) -> tuple[int, int, int]:
    records = collect_dns_records(dns)
    marker = normalize_text(domain_marker)
    domain_values = [first_question_name(dns)]
    for record in records:
        domain_values.extend([record["name"], record["value"]])
    has_domain = any(marker in value for value in domain_values if value)
    has_ip = any(attack_ip in record["value"] for record in records)
    return int(has_domain and has_ip), int(has_domain), int(has_ip)


def extract_feature_row(
    packet,
    packet_index: int,
    *,
    source_pcap_version: str,
    source_pcap_file: str,
    domain_marker: str,
    attack_ip: str,
) -> dict:
    dns = packet[DNS]
    ip = packet[IP] if packet.haslayer(IP) else None
    ipv6 = packet[IPv6] if packet.haslayer(IPv6) else None
    udp = packet[UDP] if packet.haslayer(UDP) else None
    tcp = packet[TCP] if packet.haslayer(TCP) else None
    records = collect_dns_records(dns)
    qname = first_question_name(dns)

    section_count = Counter(record["section"] for record in records)
    type_count = Counter((record["section"], record["type"]) for record in records)
    ttls = [record["ttl"] for record in records if not math.isnan(record["ttl"])]
    if ttls:
        ttl_min = min(ttls)
        ttl_max = max(ttls)
        ttl_mean = sum(ttls) / len(ttls)
        ttl_std = math.sqrt(sum((ttl - ttl_mean) ** 2 for ttl in ttls) / len(ttls))
    else:
        ttl_min = ttl_max = ttl_mean = ttl_std = -1

    answer_names = [r["name"] for r in records if r["section"] == "answer" and r["name"]]
    additional_names = [r["name"] for r in records if r["section"] == "additional" and r["name"]]
    all_names = [r["name"] for r in records if r["name"]]
    answer_ips = [
        r["value"] for r in records if r["section"] == "answer" and r["type"] in {"1", "28"}
    ]
    label, domain_match, ip_match = apply_label_rule(dns, domain_marker, attack_ip)

    src_ip = str(ip.src if ip else ipv6.src if ipv6 else "")
    dst_ip = str(ip.dst if ip else ipv6.dst if ipv6 else "")
    src_port = int(udp.sport if udp else tcp.sport if tcp else 0)
    dst_port = int(udp.dport if udp else tcp.dport if tcp else 0)
    model_features = {
        "frame_len": to_float(len(packet)),
        "frame_cap_len": to_float(len(packet)),
        "ip_len": to_float(getattr(ip, "len", None)) if ip else -1,
        "ip_ttl": to_float(getattr(ip, "ttl", None)) if ip else -1,
        "ip_proto": to_float(getattr(ip, "proto", None)) if ip else -1,
        "udp_length": to_float(getattr(udp, "len", None)) if udp else -1,
        "tcp_len": to_float(len(tcp.payload)) if tcp else -1,
        "src_port": src_port,
        "dst_port": dst_port,
        "dns_id": to_int(dns.id),
        "dns_flags_authoritative": to_int(dns.aa),
        "dns_flags_truncated": to_int(dns.tc),
        "dns_flags_recdesired": to_int(dns.rd),
        "dns_flags_recavail": to_int(dns.ra),
        "dns_flags_rcode": to_int(dns.rcode),
        "dns_count_queries": to_int(dns.qdcount),
        "dns_count_answers": to_int(dns.ancount),
        "dns_count_auth_rr": to_int(dns.nscount),
        "dns_count_add_rr": to_int(dns.arcount),
        "record_total": len(records),
        "answer_record_count": section_count["answer"],
        "authority_record_count": section_count["authority"],
        "additional_record_count": section_count["additional"],
        "answer_A_count": type_count[("answer", "1")],
        "answer_NS_count": type_count[("answer", "2")],
        "answer_CNAME_count": type_count[("answer", "5")],
        "answer_AAAA_count": type_count[("answer", "28")],
        "authority_NS_count": type_count[("authority", "2")],
        "authority_SOA_count": type_count[("authority", "6")],
        "additional_A_count": type_count[("additional", "1")],
        "additional_AAAA_count": type_count[("additional", "28")],
        "additional_NS_count": type_count[("additional", "2")],
        "additional_CNAME_count": type_count[("additional", "5")],
        "ttl_min": ttl_min,
        "ttl_max": ttl_max,
        "ttl_mean": ttl_mean,
        "ttl_std": ttl_std,
        "query_name_len": len(qname),
        "query_label_count": qname.count(".") + 1 if qname else 0,
        "query_entropy": entropy(qname),
        "unique_record_name_count": len(set(all_names)),
        "answer_matches_query_count": sum(same_or_subdomain(name, qname) for name in answer_names),
        "additional_out_of_bailiwick_count": sum(
            1 for name in additional_names if qname and not same_or_subdomain(name, qname)
        ),
        "has_answer": int(section_count["answer"] > 0),
        "has_authority": int(section_count["authority"] > 0),
        "has_additional": int(section_count["additional"] > 0),
        "has_additional_A": int(type_count[("additional", "1")] > 0),
        "has_authority_NS": int(type_count[("authority", "2")] > 0),
    }
    row = {
        "packet_index": packet_index,
        "timestamp": float(getattr(packet, "time", time.time())),
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": src_port,
        "dst_port": dst_port,
        "transport": "udp" if udp else "tcp" if tcp else "other",
        "qname": qname,
        "answer_ips": ";".join(answer_ips),
        "label": label,
        "attack_type": "dns_cache_poisoning" if label else "",
        "scenario_tag": "content_rule_mixed_pcap",
        **model_features,
        "is_response": int(dns.qr or 0),
        "opcode": int(dns.opcode or 0),
        "rcode": int(dns.rcode or 0),
        "structural_rr_count": int((dns.ancount or 0) + (dns.nscount or 0) + (dns.arcount or 0)),
        "min_ttl": ttl_min,
        "max_ttl": ttl_max,
        "source_pcap_version": source_pcap_version,
        "source_pcap_file": source_pcap_file,
        "label_domain_match": domain_match,
        "label_ip_match": ip_match,
        "label_rule": f"contains:{domain_marker}&rr_contains:{attack_ip}",
    }
    return {column: row.get(column) for column in CSV_COLUMNS}
