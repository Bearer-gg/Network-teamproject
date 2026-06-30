"""Feature and artifact schema owned by the local ML pipeline."""

LABEL_COLUMN = "label"

META_COLUMNS = [
    "packet_index",
    "timestamp",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "transport",
    "qname",
    "answer_ips",
    "dns_id",
    "label",
    "attack_type",
    "scenario_tag",
]

FEATURE_COLUMNS = [
    "frame_len",
    "frame_cap_len",
    "ip_len",
    "ip_ttl",
    "ip_proto",
    "udp_length",
    "tcp_len",
    "src_port",
    "dst_port",
    "dns_id",
    "dns_flags_authoritative",
    "dns_flags_truncated",
    "dns_flags_recdesired",
    "dns_flags_recavail",
    "dns_flags_rcode",
    "dns_count_queries",
    "dns_count_answers",
    "dns_count_auth_rr",
    "dns_count_add_rr",
    "record_total",
    "answer_record_count",
    "authority_record_count",
    "additional_record_count",
    "answer_A_count",
    "answer_NS_count",
    "answer_CNAME_count",
    "answer_AAAA_count",
    "authority_NS_count",
    "authority_SOA_count",
    "additional_A_count",
    "additional_AAAA_count",
    "additional_NS_count",
    "additional_CNAME_count",
    "ttl_min",
    "ttl_max",
    "ttl_mean",
    "ttl_std",
    "query_name_len",
    "query_label_count",
    "query_entropy",
    "unique_record_name_count",
    "answer_matches_query_count",
    "additional_out_of_bailiwick_count",
    "has_answer",
    "has_authority",
    "has_additional",
    "has_additional_A",
    "has_authority_NS",
]

IDS_COMPAT_COLUMNS = [
    "is_response",
    "opcode",
    "rcode",
    "structural_rr_count",
    "min_ttl",
    "max_ttl",
]

PROVENANCE_COLUMNS = [
    "source_pcap_version",
    "source_pcap_file",
    "label_domain_match",
    "label_ip_match",
    "label_rule",
]

CSV_COLUMNS = (
    META_COLUMNS
    + [column for column in FEATURE_COLUMNS if column not in META_COLUMNS]
    + IDS_COMPAT_COLUMNS
    + PROVENANCE_COLUMNS
)
