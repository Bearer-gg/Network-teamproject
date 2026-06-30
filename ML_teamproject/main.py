#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from pipeline.dataset import build_dataset
from pipeline.feature_experiments import run_feature_experiments, write_feature_experiment_notebook
from pipeline.training import build_run_input_manifest, compare_threshold_weight_grid, execute_training_notebook
from pipeline.transfer import deploy_model, fetch_data
from pipeline.versioning import (
    next_run_version,
    normalize_run_version,
    normalize_version,
    resolve_version,
    run_bundle_dir,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def _capture_scope_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--capture-scope",
        choices=["resolver-bound", "all-dns-responses"],
        default="resolver-bound",
        help="Use all-dns-responses for external pcaps without internal resolver/port constraints",
    )


def add_connection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", required=True, help="Ubuntu host or IP address")
    parser.add_argument("--username", help="SSH username")
    parser.add_argument("--identity-file", help="SSH private key file")
    parser.add_argument("--port", type=int, help="SSH port")


def _dataset_version_args(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    parser.add_argument("--dataset-version", action="append", required=required)
    parser.add_argument("--dataset-versions", help="Space or comma separated dataset versions")


def _scenario_dataset_version_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-dataset-version", action="append")
    parser.add_argument("--train-dataset-versions", help="Space or comma separated train scenario versions")
    parser.add_argument("--test-dataset-version", action="append")
    parser.add_argument("--test-dataset-versions", help="Space or comma separated test scenario versions")


def _parsed_versions(args: argparse.Namespace, singular: str, plural: str) -> list[str]:
    raw_values: list[str] = []
    if getattr(args, singular, None):
        raw_values.extend(getattr(args, singular))
    if getattr(args, plural, None):
        raw_values.append(getattr(args, plural))
    values = [
        item
        for raw_value in raw_values
        for item in str(raw_value).replace(",", " ").split()
        if item.strip()
    ]
    return [normalize_version(value) for value in values]


def _parse_dataset_versions(args: argparse.Namespace) -> list[str]:
    values = _parsed_versions(args, "dataset_version", "dataset_versions")
    if not values:
        raise ValueError("At least one dataset version is required.")
    return values


def _parse_scenario_split(args: argparse.Namespace) -> tuple[list[str] | None, list[str] | None]:
    train_versions = _parsed_versions(args, "train_dataset_version", "train_dataset_versions")
    test_versions = _parsed_versions(args, "test_dataset_version", "test_dataset_versions")
    if train_versions or test_versions:
        if not train_versions or not test_versions:
            raise ValueError("Scenario split requires both train and test dataset versions.")
        return train_versions, test_versions
    return None, None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local RandomForest pipeline for DNS cache poisoning pcap datasets."
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Build one dataset and train one run")
    run_parser.add_argument("--pcap", required=True)
    run_parser.add_argument("--resolver-ip")
    _capture_scope_args(run_parser)
    run_parser.add_argument("--run-version", help="Defaults to the next available run version")
    run_parser.add_argument("--dataset-version")
    run_parser.add_argument("--config", default="config/randomforest.yaml")
    run_parser.add_argument("--overwrite", action="store_true")

    build_parser_command = subparsers.add_parser("build-dataset", help="Convert one pcap to a labeled CSV")
    build_parser_command.add_argument("--pcap", required=True)
    build_parser_command.add_argument("--resolver-ip")
    _capture_scope_args(build_parser_command)
    build_parser_command.add_argument("--dataset-version")
    build_parser_command.add_argument("--version")
    build_parser_command.add_argument("--overwrite", action="store_true")

    train_parser = subparsers.add_parser("train", help="Execute the local training notebook")
    train_parser.add_argument("--run-version", help="Defaults to the next available run version")
    _dataset_version_args(train_parser, required=False)
    _scenario_dataset_version_args(train_parser)
    train_parser.add_argument("--run-manifest")
    train_parser.add_argument("--config", default="config/randomforest.yaml")
    train_parser.add_argument("--overwrite", action="store_true")

    prepare_parser = subparsers.add_parser("prepare-run", help="Create a run input manifest")
    prepare_parser.add_argument("--run-version", help="Defaults to the next available run version")
    _dataset_version_args(prepare_parser, required=False)
    _scenario_dataset_version_args(prepare_parser)
    prepare_parser.add_argument("--config", default="config/randomforest.yaml")
    prepare_parser.add_argument("--overwrite", action="store_true")

    compare_parser = subparsers.add_parser(
        "compare-thresholds",
        help="Compare confusion matrices across decision thresholds and class weights",
    )
    compare_parser.add_argument("--run-version", required=True)
    compare_parser.add_argument("--analysis-name")
    compare_parser.add_argument("--config", default="config/randomforest.yaml")
    compare_parser.add_argument("--overwrite", action="store_true")

    feature_parser = subparsers.add_parser(
        "feature-experiments",
        help="Run fixed-threshold feature-set experiments for an existing trained run",
    )
    feature_parser.add_argument("--run-version", required=True)
    feature_parser.add_argument("--config", default="config/randomforest.yaml")
    feature_parser.add_argument("--output-root", default="outputs/feature_experiments")
    feature_parser.add_argument("--max-plotted-features", type=int, default=12)
    feature_parser.add_argument("--overwrite", action="store_true")

    feature_report_parser = subparsers.add_parser(
        "feature-report",
        help="Build a notebook report from an existing feature experiment output directory",
    )
    feature_report_parser.add_argument("--run-version", required=True)
    feature_report_parser.add_argument("--output-root", default="outputs/feature_experiments")

    fetch_parser = subparsers.add_parser("fetch-data", help="Retrieve a versioned Ubuntu pcap via scp")
    fetch_parser.add_argument("--dataset-version", required=True)
    fetch_parser.add_argument("--version")
    fetch_parser.add_argument("--remote-capture-dir", required=True)
    fetch_parser.add_argument("--overwrite", action="store_true")
    add_connection_arguments(fetch_parser)

    deploy_parser = subparsers.add_parser("deploy-model", help="Upload a selected versioned model via scp")
    deploy_parser.add_argument("--run-version", required=True)
    deploy_parser.add_argument("--version")
    deploy_parser.add_argument("--remote-model-dir", required=True)
    deploy_parser.add_argument("--overwrite", action="store_true")
    add_connection_arguments(deploy_parser)
    return parser


def _confirm(message: str) -> bool:
    try:
        return input(f"{message} [y/N]: ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def _run_model_path(run_version: str) -> Path:
    return run_bundle_dir(PROJECT_ROOT, run_version) / f"randomforest_{normalize_run_version(run_version)}.joblib"


def _resolved_run_version(value: str | None, run_manifest: str | None = None) -> str:
    if run_manifest:
        manifest_path = Path(run_manifest)
        with manifest_path.open(encoding="utf-8") as handle:
            manifest_version = normalize_run_version(yaml.safe_load(handle)["run_version"])
        if value and normalize_run_version(value) != manifest_version:
            raise ValueError(
                f"Run version {normalize_run_version(value)} does not match manifest run version {manifest_version}."
            )
        return manifest_version
    return normalize_run_version(value) if value else next_run_version(PROJECT_ROOT)


def _deployment_prompt(run_version: str) -> None:
    if not _confirm("Deploy this trained model to Ubuntu now?"):
        print("Model was not deployed.")
        return
    host = input("Ubuntu host or IP: ").strip()
    username = input("SSH username (blank for current user): ").strip() or None
    identity_file = input("SSH identity file (blank for SSH default): ").strip() or None
    remote_model_dir = input("Remote model directory: ").strip()
    overwrite = _confirm("Allow replacing the same remote model filename if it already exists?")
    remote_path = deploy_model(
        model_path=_run_model_path(run_version),
        host=host,
        username=username,
        identity_file=identity_file,
        remote_model_dir=remote_model_dir,
        overwrite=overwrite,
    )
    print(f"Uploaded model: {remote_path}")
    print("Restart the sniffer to select the highest numbered uploaded model.")


def run_pipeline(args: argparse.Namespace) -> None:
    dataset_version = normalize_version(
        args.dataset_version or resolve_version(args.pcap, getattr(args, "version", None))
    )
    run_version = _resolved_run_version(args.run_version)
    print(f"Run version: {run_version}")
    dataset_result = build_dataset(
        pcap_path=args.pcap,
        resolver_ip=args.resolver_ip,
        capture_scope=args.capture_scope,
        version=dataset_version,
        project_root=PROJECT_ROOT,
        overwrite=args.overwrite,
    )
    print(f"Wrote dataset: {dataset_result['csv_path']}")
    print(f"Accepted rows: {dataset_result['accepted_rows']}; labels: {dataset_result['label_counts']}")
    run_manifest_result = build_run_input_manifest(
        dataset_versions=[dataset_version],
        run_version=run_version,
        project_root=PROJECT_ROOT,
        config_path=args.config,
        overwrite=args.overwrite,
    )
    executed = execute_training_notebook(
        run_version=run_version,
        run_manifest_path=run_manifest_result["manifest_path"],
        project_root=PROJECT_ROOT,
        config_path=args.config,
        overwrite=args.overwrite,
    )
    print(f"Executed notebook: {executed}")
    print(f"Trained model: {_run_model_path(run_version)}")
    _deployment_prompt(run_version)


def interactive_run() -> None:
    print("Local DNS Cache Poisoning RandomForest Pipeline")
    print("==============================================")
    local_pcap = input("Local pcap path (leave blank to fetch from Ubuntu): ").strip()
    dataset_version = input("Dataset version (for example v000001): ").strip()
    run_version = input("Run version (blank for next available version): ").strip()
    if not local_pcap:
        host = input("Ubuntu host or IP: ").strip()
        username = input("SSH username (blank for current user): ").strip() or None
        identity_file = input("SSH identity file (blank for SSH default): ").strip() or None
        remote_capture_dir = input("Remote capture directory: ").strip()
        path = fetch_data(
            dataset_version=dataset_version,
            host=host,
            username=username,
            identity_file=identity_file,
            remote_capture_dir=remote_capture_dir,
            local_raw_dir=PROJECT_ROOT / "data" / "raw",
        )
        local_pcap = str(path)
    resolver_ip = input("Resolver destination IP used in the pcap: ").strip()
    args = argparse.Namespace(
        pcap=local_pcap,
        resolver_ip=resolver_ip,
        dataset_version=dataset_version or None,
        run_version=run_version or None,
        version=None,
        config="config/randomforest.yaml",
        capture_scope="resolver-bound",
        overwrite=False,
    )
    run_pipeline(args)


def main() -> None:
    args = build_parser().parse_args()
    if not args.command:
        interactive_run()
    elif args.command == "run":
        run_pipeline(args)
    elif args.command == "build-dataset":
        dataset_version = args.dataset_version or args.version
        if not dataset_version:
            dataset_version = None
        result = build_dataset(
            pcap_path=args.pcap,
            resolver_ip=args.resolver_ip,
            capture_scope=args.capture_scope,
            version=dataset_version,
            project_root=PROJECT_ROOT,
            overwrite=args.overwrite,
        )
        print(f"Wrote dataset: {result['csv_path']}")
        print(f"Accepted rows: {result['accepted_rows']}; labels: {result['label_counts']}")
    elif args.command == "train":
        run_version = _resolved_run_version(args.run_version, args.run_manifest)
        print(f"Run version: {run_version}")
        if args.run_manifest:
            run_manifest_result = {"manifest_path": Path(args.run_manifest)}
        else:
            train_dataset_versions, test_dataset_versions = _parse_scenario_split(args)
            dataset_versions = None if train_dataset_versions else _parse_dataset_versions(args)
            run_manifest_result = build_run_input_manifest(
                dataset_versions=dataset_versions,
                train_dataset_versions=train_dataset_versions,
                test_dataset_versions=test_dataset_versions,
                run_version=run_version,
                project_root=PROJECT_ROOT,
                config_path=args.config,
                overwrite=args.overwrite,
            )
        executed = execute_training_notebook(
            run_version=run_version,
            run_manifest_path=run_manifest_result["manifest_path"],
            project_root=PROJECT_ROOT,
            config_path=args.config,
            overwrite=args.overwrite,
        )
        print(f"Executed notebook: {executed}")
        print(f"Run bundle: {run_bundle_dir(PROJECT_ROOT, run_version)}")
    elif args.command == "prepare-run":
        run_version = _resolved_run_version(args.run_version)
        print(f"Run version: {run_version}")
        train_dataset_versions, test_dataset_versions = _parse_scenario_split(args)
        dataset_versions = None if train_dataset_versions else _parse_dataset_versions(args)
        result = build_run_input_manifest(
            dataset_versions=dataset_versions,
            train_dataset_versions=train_dataset_versions,
            test_dataset_versions=test_dataset_versions,
            run_version=run_version,
            project_root=PROJECT_ROOT,
            config_path=args.config,
            overwrite=args.overwrite,
        )
        print(f"Run input manifest: {result['manifest_path']}")
    elif args.command == "compare-thresholds":
        result = compare_threshold_weight_grid(
            run_version=args.run_version,
            project_root=PROJECT_ROOT,
            config_path=args.config,
            analysis_name=args.analysis_name,
            overwrite=args.overwrite,
        )
        print(f"Comparison CSV: {result['csv_path']}")
        print(f"Analysis manifest: {result['manifest_path']}")
    elif args.command == "feature-experiments":
        result = run_feature_experiments(
            run_version=args.run_version,
            project_root=PROJECT_ROOT,
            config_path=args.config,
            output_root=args.output_root,
            overwrite=args.overwrite,
            max_plotted_features=args.max_plotted_features,
        )
        print(f"Feature experiment output: {result['output_dir']}")
        print(f"Metrics CSV: {result['metrics_path']}")
        print(f"Statistics CSV: {result['statistics_path']}")
        print(f"Report notebook: {result['notebook_path']}")
        print(f"Plots: {result['plots_dir']}")
    elif args.command == "feature-report":
        output_dir = PROJECT_ROOT / args.output_root / f"run_{normalize_run_version(args.run_version)}"
        notebook_path = write_feature_experiment_notebook(output_dir)
        print(f"Report notebook: {notebook_path}")
    elif args.command == "fetch-data":
        dataset_version = normalize_version(args.dataset_version or args.version)
        output = fetch_data(
            dataset_version=dataset_version,
            host=args.host,
            username=args.username,
            identity_file=args.identity_file,
            port=args.port,
            remote_capture_dir=args.remote_capture_dir,
            local_raw_dir=PROJECT_ROOT / "data" / "raw",
            overwrite=args.overwrite,
        )
        print(f"Downloaded pcap: {output}")
    elif args.command == "deploy-model":
        run_version = normalize_run_version(args.run_version or args.version)
        model = _run_model_path(run_version)
        if not _confirm(f"Upload selected model {model.name} to Ubuntu?"):
            print("Deployment cancelled.")
            return
        remote_path = deploy_model(
            model_path=model,
            host=args.host,
            username=args.username,
            identity_file=args.identity_file,
            port=args.port,
            remote_model_dir=args.remote_model_dir,
            overwrite=args.overwrite,
        )
        print(f"Uploaded model: {remote_path}")
        print("Restart the sniffer to select the highest numbered uploaded model.")


if __name__ == "__main__":
    main()
