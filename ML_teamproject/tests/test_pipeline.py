from __future__ import annotations

import json
import shutil
import csv
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd
import pytest
from scapy.all import DNS, DNSQR, DNSRR, IP, UDP, wrpcap

from main import _parse_scenario_split
from pipeline.dataset import build_dataset
from pipeline.feature_experiments import (
    build_feature_sets,
    run_feature_experiments,
    write_feature_experiment_notebook,
)
from pipeline.features import apply_label_rule
from pipeline.schema import FEATURE_COLUMNS
from pipeline.training import (
    build_run_input_manifest,
    compare_threshold_weight_grid,
    execute_training_notebook,
    load_official_model,
    train_model,
    train_scratch_model,
)
from pipeline.transfer import deploy_model, fetch_data
from pipeline.versioning import next_run_version, normalize_version, run_bundle_dir, version_from_pcap


ATTACK_IP = "192.168.219.104"


def dns_response(qname: str, rrname: str, rdata: str):
    return DNS(
        qr=1,
        qd=DNSQR(qname=qname),
        ar=DNSRR(rrname=rrname, type="A", rdata=rdata),
        arcount=1,
    )


def captured_packet(dns):
    return IP(src="192.168.219.2", dst="192.168.219.1") / UDP(sport=20053, dport=10053) / dns


def write_training_inputs(root: Path, dataset_version: str, labels: list[int] | None = None) -> Path:
    processed = root / "data" / "processed" / f"features_{dataset_version}.csv"
    processed.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    labels = labels or [index % 2 for index in range(20)]
    for index, label in enumerate(labels):
        row = {column: float(index + label) for column in FEATURE_COLUMNS}
        row.update({"label": label, "source_dataset_version": dataset_version, "source_processed_csv": str(processed)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(processed, index=False)
    config = root / "config" / "randomforest.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        "\n".join(
            [
                "random_seed: 42",
                "test_size: 0.2",
                "search_iterations: 1",
                "cv_folds: 2",
                "search_n_jobs: 1",
                "model_n_jobs: 1",
                "scoring: f1",
                "selected_feature_columns: []",
                "parameter_distributions:",
                "  n_estimators: [5]",
                "  max_depth: [null]",
                "  min_samples_split: [2]",
                "  min_samples_leaf: [1]",
                "  max_features: [sqrt]",
                "  class_weight: [balanced]",
            ]
        ),
        encoding="utf-8",
    )
    return processed


def test_label_requires_domain_and_attack_ip_in_dns_fields():
    both = apply_label_rule(dns_response("other.test", "ns1.bank.test", ATTACK_IP), "bank.test", ATTACK_IP)
    domain_only = apply_label_rule(dns_response("www.bank.test", "ns1.bank.test", "1.1.1.1"), "bank.test", ATTACK_IP)
    ip_only = apply_label_rule(dns_response("other.test", "ns.other.test", ATTACK_IP), "bank.test", ATTACK_IP)
    assert both == (1, 1, 1)
    assert domain_only == (0, 1, 0)
    assert ip_only == (0, 0, 1)


def test_build_dataset_labels_rows_inside_one_pcap(tmp_path: Path):
    source = tmp_path / "dataset_v000003.pcap"
    packets = [
        captured_packet(dns_response("www.bank.test", "www.bank.test", ATTACK_IP)),
        captured_packet(dns_response("www.bank.test", "www.bank.test", "8.8.8.8")),
        IP(src="192.168.219.2", dst="10.0.0.1")
        / UDP(sport=20053, dport=10053)
        / dns_response("www.bank.test", "www.bank.test", ATTACK_IP),
    ]
    wrpcap(str(source), packets)
    result = build_dataset(
        pcap_path=source,
        resolver_ip="192.168.219.1",
        project_root=tmp_path,
    )
    frame = pd.read_csv(result["csv_path"])
    assert result["accepted_rows"] == 2
    assert frame["label"].tolist() == [1, 0]
    assert list(frame[FEATURE_COLUMNS].columns) == FEATURE_COLUMNS


def test_external_dns_scope_accepts_standard_dns_response_without_internal_resolver(tmp_path: Path):
    source = tmp_path / "external_dns.pcap"
    packet = (
        IP(src="8.8.8.8", dst="10.0.0.34")
        / UDP(sport=53, dport=55012)
        / dns_response("example.test", "example.test", "1.1.1.1")
    )
    wrpcap(str(source), [packet])
    result = build_dataset(
        pcap_path=source,
        resolver_ip=None,
        capture_scope="all-dns-responses",
        version="v000099",
        project_root=tmp_path,
    )
    frame = pd.read_csv(result["csv_path"])
    assert result["accepted_rows"] == 1
    assert frame["label"].tolist() == [0]


def test_training_artifact_uses_feature_contract(tmp_path: Path):
    processed = write_training_inputs(tmp_path, "v000004")
    result = train_model(
        dataset_versions=["v000004"],
        run_version="v000004",
        project_root=tmp_path,
        config_path=tmp_path / "config" / "randomforest.yaml",
    )
    artifact = joblib.load(result["paths"]["model"])
    assert artifact["feature_columns"] == FEATURE_COLUMNS
    assert set(artifact) >= {"model", "feature_columns", "best_params"}
    metrics = json.loads(result["paths"]["metrics"].read_text(encoding="utf-8"))
    assert metrics["evaluation_strategy"] == "stratified_random_row_split_across_selected_datasets"
    assert result["paths"]["run_dir"] == run_bundle_dir(tmp_path, "v000004")


def test_notebook_is_executed_locally_and_writes_artifact(tmp_path: Path):
    processed = write_training_inputs(tmp_path, "v000005")
    notebook_dir = tmp_path / "notebook"
    notebook_dir.mkdir()
    shutil.copy(Path("notebook/detection.ipynb"), notebook_dir / "detection.ipynb")
    run_path = execute_training_notebook(
        dataset_versions=["v000005"],
        run_version="v000005",
        project_root=tmp_path,
        config_path=tmp_path / "config" / "randomforest.yaml",
    )
    notebook_text = run_path.read_text(encoding="utf-8")
    assert "Random Forest Training Summary" in notebook_text
    assert (tmp_path / "runs" / "run_v000005" / "randomforest_v000005.joblib").exists()


def test_training_can_use_multiple_datasets_for_one_run(tmp_path: Path):
    write_training_inputs(tmp_path, "v000010", labels=[0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
    write_training_inputs(tmp_path, "v000011", labels=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    manifest_result = build_run_input_manifest(
        dataset_versions=["v000010", "v000011"],
        run_version="v000020",
        project_root=tmp_path,
        config_path=tmp_path / "config" / "randomforest.yaml",
    )
    result = train_model(
        run_version="v000020",
        run_manifest_path=manifest_result["manifest_path"],
        project_root=tmp_path,
        config_path=tmp_path / "config" / "randomforest.yaml",
    )
    metrics = json.loads(result["paths"]["metrics"].read_text(encoding="utf-8"))
    assert metrics["dataset_versions"] == ["v000010", "v000011"]
    assert len(metrics["source_processed_csv_files"]) == 2
    assert (tmp_path / "runs" / "run_v000020" / "randomforest_v000020.joblib").exists()
    assert manifest_result["manifest"]["dataset_versions"] == ["v000010", "v000011"]


def test_scenario_split_uses_separate_train_and_test_datasets(tmp_path: Path):
    write_training_inputs(tmp_path, "v000012", labels=[0, 1] * 10)
    write_training_inputs(tmp_path, "v000013", labels=[0, 0, 1, 1] * 3)
    manifest_result = build_run_input_manifest(
        train_dataset_versions=["v000012"],
        test_dataset_versions=["v000013"],
        run_version="v000021",
        project_root=tmp_path,
        config_path=tmp_path / "config" / "randomforest.yaml",
    )
    result = train_model(
        run_version="v000021",
        run_manifest_path=manifest_result["manifest_path"],
        project_root=tmp_path,
        config_path=tmp_path / "config" / "randomforest.yaml",
    )
    metrics = result["metrics"]
    assert metrics["evaluation_strategy"] == "scenario_split_by_dataset_version"
    assert metrics["train_dataset_versions"] == ["v000012"]
    assert metrics["test_dataset_versions"] == ["v000013"]
    assert metrics["train_row_count"] == 20
    assert metrics["test_row_count"] == 12
    assert manifest_result["manifest"]["evaluation_strategy"] == "scenario_split_by_dataset_version"


def test_scenario_split_combines_multiple_datasets_and_applies_configured_ratios(tmp_path: Path):
    write_training_inputs(tmp_path, "v000070", labels=[0] * 8 + [1] * 8)
    write_training_inputs(tmp_path, "v000071", labels=[0] * 4 + [1] * 4)
    write_training_inputs(tmp_path, "v000072", labels=[0] * 5 + [1] * 10)
    write_training_inputs(tmp_path, "v000073", labels=[0] * 7 + [1] * 2)
    config = tmp_path / "config" / "randomforest.yaml"
    config.write_text(
        config.read_text(encoding="utf-8")
        .replace("selected_feature_columns: []", 'class_ratio:\n  train: "3:1"\n  test: "1:1"\nselected_feature_columns: []'),
        encoding="utf-8",
    )
    result = train_scratch_model(
        train_dataset_versions=["v000070", "v000071"],
        test_dataset_versions=["v000072", "v000073"],
        scratch_name="combined_ratio",
        project_root=tmp_path,
        config_path=config,
    )
    metrics = result["metrics"]
    assert metrics["train_dataset_versions"] == ["v000070", "v000071"]
    assert metrics["test_dataset_versions"] == ["v000072", "v000073"]
    assert metrics["class_ratio"] == {
        "train": {"normal": 3, "attack": 1},
        "test": {"normal": 1, "attack": 1},
    }
    assert metrics["class_ratio_sampling_train"]["class_distribution_before"] == {"0": 12, "1": 12}
    assert metrics["class_distribution_train"] == {"0": 12, "1": 4}
    assert metrics["class_ratio_sampling_test"]["class_distribution_before"] == {"0": 12, "1": 12}
    assert metrics["class_distribution_test"] == {"0": 12, "1": 12}


def test_scenario_split_refuses_dataset_overlap(tmp_path: Path):
    write_training_inputs(tmp_path, "v000014")
    with pytest.raises(ValueError, match="must not overlap"):
        train_scratch_model(
            train_dataset_versions=["v000014"],
            test_dataset_versions=["v000014"],
            scratch_name="leaking_split",
            project_root=tmp_path,
            config_path=tmp_path / "config" / "randomforest.yaml",
        )


def test_scenario_split_accepts_multiple_versions_in_singular_cli_option():
    args = type(
        "Args",
        (),
        {
            "train_dataset_version": ["v000001 v000004"],
            "train_dataset_versions": None,
            "test_dataset_version": ["v000003"],
            "test_dataset_versions": None,
        },
    )()
    assert _parse_scenario_split(args) == (["v000001", "v000004"], ["v000003"])


def test_run_input_manifest_records_dataset_lineage(tmp_path: Path):
    write_training_inputs(tmp_path, "v000040")
    result = build_run_input_manifest(
        dataset_versions=["v000040"],
        run_version="v000040",
        project_root=tmp_path,
        config_path=tmp_path / "config" / "randomforest.yaml",
    )
    manifest = result["manifest"]
    assert manifest["dataset_versions"] == ["v000040"]
    assert len(manifest["source_processed_csv_checksums"]) == 1
    assert result["manifest_path"].exists()


def test_training_can_restrict_selected_features(tmp_path: Path):
    processed = write_training_inputs(tmp_path, "v000030")
    config = tmp_path / "config" / "randomforest.yaml"
    config.write_text(
        "\n".join(
            [
                "random_seed: 42",
                "test_size: 0.2",
                "search_iterations: 1",
                "cv_folds: 2",
                "search_n_jobs: 1",
                "model_n_jobs: 1",
                "scoring: f1",
                "selected_feature_columns:",
                "  - frame_len",
                "  - dns_id",
                "  - ttl_max",
                "parameter_distributions:",
                "  n_estimators: [5]",
                "  max_depth: [null]",
                "  min_samples_split: [2]",
                "  min_samples_leaf: [1]",
                "  max_features: [sqrt]",
                "  class_weight: [balanced]",
            ]
        ),
        encoding="utf-8",
    )
    result = train_model(
        dataset_versions=["v000030"],
        run_version="v000030",
        project_root=tmp_path,
        config_path=config,
    )
    artifact = joblib.load(result["paths"]["model"])
    assert artifact["feature_columns"] == ["frame_len", "dns_id", "ttl_max"]


def test_training_excludes_configured_feature_from_default_set(tmp_path: Path):
    write_training_inputs(tmp_path, "v000031")
    config = tmp_path / "config" / "randomforest.yaml"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "selected_feature_columns: []",
            "selected_feature_columns: []\nexcluded_feature_columns:\n  - dns_id",
        ),
        encoding="utf-8",
    )
    result = train_model(
        dataset_versions=["v000031"],
        run_version="v000031",
        project_root=tmp_path,
        config_path=config,
    )
    artifact = joblib.load(result["paths"]["model"])
    assert "dns_id" not in artifact["feature_columns"]
    assert artifact["excluded_feature_columns"] == ["dns_id"]


def test_feature_contract_has_no_time_or_capture_order_columns():
    forbidden_fragments = (
        "time",
        "timestamp",
        "delta",
        "per_second",
        "packet_index",
        "capture",
        "order",
    )
    assert not [
        column
        for column in FEATURE_COLUMNS
        if any(fragment in column for fragment in forbidden_fragments)
    ]


def test_summary_comment_column_survives_overwrite(tmp_path: Path):
    write_training_inputs(tmp_path, "v000050")
    kwargs = {
        "dataset_versions": ["v000050"],
        "run_version": "v000050",
        "project_root": tmp_path,
        "config_path": tmp_path / "config" / "randomforest.yaml",
    }
    result = train_model(**kwargs)
    summary_path = result["paths"]["summary"]
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys())
    rows[0]["comment"] = "manual note"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    train_model(**kwargs, overwrite=True)
    with summary_path.open(newline="", encoding="utf-8") as handle:
        overwritten = list(csv.DictReader(handle))
    assert overwritten[0]["comment"] == "manual note"


def test_distinct_run_appends_to_existing_summary_without_overwrite(tmp_path: Path):
    write_training_inputs(tmp_path, "v000051")
    kwargs = {
        "dataset_versions": ["v000051"],
        "project_root": tmp_path,
        "config_path": tmp_path / "config" / "randomforest.yaml",
    }
    train_model(**kwargs, run_version="v000051")
    train_model(**kwargs, run_version="v000052")
    with (tmp_path / "metrics" / "metrics_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["run_version"] for row in rows] == ["v000051", "v000052"]


def test_scratch_training_writes_only_to_scratch_directory(tmp_path: Path):
    write_training_inputs(tmp_path, "v000060")
    result = train_scratch_model(
        dataset_versions=["v000060"],
        scratch_name="feature_try_01",
        project_root=tmp_path,
        config_path=tmp_path / "config" / "randomforest.yaml",
        feature_columns=["frame_len", "dns_id"],
    )
    artifact = joblib.load(result["paths"]["model"])
    assert result["paths"]["model"] == tmp_path / "scratch" / "feature_try_01" / "randomforest_scratch.joblib"
    assert artifact["output_mode"] == "scratch"
    assert artifact["run_version"] is None
    assert artifact["feature_columns"] == ["frame_len", "dns_id"]
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "metrics" / "metrics_summary.csv").exists()


def test_official_model_can_be_loaded_for_read_only_manual_review(tmp_path: Path):
    write_training_inputs(tmp_path, "v000061")
    train_model(
        dataset_versions=["v000061"],
        run_version="v000061",
        project_root=tmp_path,
        config_path=tmp_path / "config" / "randomforest.yaml",
    )
    artifact = load_official_model("v000061", project_root=tmp_path)
    assert artifact["run_version"] == "v000061"
    assert artifact["output_mode"] == "official"


def test_threshold_weight_comparison_writes_scratch_table(tmp_path: Path):
    write_training_inputs(tmp_path, "v000062")
    config = tmp_path / "config" / "randomforest.yaml"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "selected_feature_columns: []",
            "decision_thresholds: [0.25, 0.5]\nclass_weight_grid:\n  - null\n  - {0: 1, 1: 3}\nselected_feature_columns: []",
        ),
        encoding="utf-8",
    )
    train_model(
        dataset_versions=["v000062"],
        run_version="v000062",
        project_root=tmp_path,
        config_path=config,
    )
    result = compare_threshold_weight_grid(
        run_version="v000062",
        analysis_name="threshold_check",
        project_root=tmp_path,
        config_path=config,
    )
    frame = pd.read_csv(result["csv_path"])
    assert set(frame["threshold"]) == {0.25, 0.5}
    assert {"base_run_model", "class_weight_none", "class_weight_0-1_1-3"}.issubset(
        set(frame["model_name"])
    )
    assert not (tmp_path / "metrics" / "metrics_summary.csv").read_text(encoding="utf-8").count("threshold_check")


def test_feature_experiments_write_analysis_bundle(tmp_path: Path):
    write_training_inputs(tmp_path, "v000080", labels=[0, 1] * 12)
    write_training_inputs(tmp_path, "v000081", labels=[0, 0, 1, 1] * 6)
    train_model(
        train_dataset_versions=["v000080"],
        test_dataset_versions=["v000081"],
        run_version="v000080",
        project_root=tmp_path,
        config_path=tmp_path / "config" / "randomforest.yaml",
    )
    result = run_feature_experiments(
        run_version="v000080",
        project_root=tmp_path,
        config_path=tmp_path / "config" / "randomforest.yaml",
        max_plotted_features=2,
    )
    metrics = pd.read_csv(result["metrics_path"])
    assert {"all_current", "dns_structure_only", "shortcut_suspects_only"}.issubset(
        set(metrics["feature_set"])
    )
    assert (result["output_dir"] / "config.yaml").exists()
    assert (result["confusion_path"]).exists()
    assert (result["importance_path"]).exists()
    assert (result["statistics_path"]).exists()
    assert (result["probability_path"]).exists()
    assert (result["notebook_path"]).exists()
    assert (result["plots_dir"] / "metrics_comparison.png").exists()
    assert (result["plots_dir"] / "probability_distribution.png").exists()
    rebuilt = write_feature_experiment_notebook(result["output_dir"])
    assert rebuilt == result["notebook_path"]


def test_feature_set_builder_allows_exceptional_shortcut_addbacks():
    sets = build_feature_sets(["dns_id", "has_additional_A", "record_total"])
    assert "all_current" in sets
    assert sets["all_current"] == ["dns_id", "has_additional_A", "record_total"]
    assert sets["add_back_dropped__frame_len"] == [
        "dns_id",
        "has_additional_A",
        "record_total",
        "frame_len",
    ]
    assert "dropped_shortcut_only" in sets


def test_version_validation():
    assert normalize_version("3") == "v000003"
    assert version_from_pcap("dataset_v000003.pcap") == "v000003"


def test_next_run_version_uses_highest_existing_run_directory(tmp_path: Path):
    (tmp_path / "runs" / "run_v000001").mkdir(parents=True)
    (tmp_path / "runs" / "run_v000009").mkdir()
    (tmp_path / "runs" / "notes").mkdir()
    assert next_run_version(tmp_path) == "v000010"


def test_fetch_refuses_existing_local_file_without_transfer(tmp_path: Path):
    existing = tmp_path / "dataset_v000001.pcap"
    existing.write_bytes(b"exists")
    with patch("pipeline.transfer.subprocess.run") as run:
        with pytest.raises(FileExistsError):
            fetch_data(
                dataset_version="v000001",
                host="example.invalid",
                remote_capture_dir="/data/captures",
                local_raw_dir=tmp_path,
            )
    run.assert_not_called()


def test_deploy_refuses_existing_remote_model_without_overwrite(tmp_path: Path):
    model = tmp_path / "runs" / "run_v000001" / "randomforest_v000001.joblib"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"model")
    with patch("pipeline.transfer.subprocess.run") as run:
        run.return_value.returncode = 0
        with pytest.raises(FileExistsError):
            deploy_model(
                model_path=model,
                host="example.invalid",
                remote_model_dir="/models",
            )
    assert run.call_count == 1
    assert run.call_args.args[0][0] == "ssh"
