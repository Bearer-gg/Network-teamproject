from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from .schema import FEATURE_COLUMNS, LABEL_COLUMN
from .versioning import dataset_version_from_features, normalize_run_version, run_bundle_dir


FEATURES_FILE_PATTERN = re.compile(r"^features_(v\d{6})\.csv$")
SCRATCH_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")


def load_config(config_path: str | Path) -> dict:
    with Path(config_path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def schema_checksum() -> str:
    payload = json.dumps(FEATURE_COLUMNS, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_checksum(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def excluded_feature_columns(config: dict) -> list[str]:
    return [str(column) for column in (config.get("excluded_feature_columns") or [])]


def selected_feature_columns(config: dict) -> list[str]:
    selected = config.get("selected_feature_columns") or []
    excluded = set(excluded_feature_columns(config))
    if not selected:
        return [column for column in FEATURE_COLUMNS if column not in excluded]
    selected = [str(column) for column in selected]
    unknown = [column for column in selected if column not in FEATURE_COLUMNS]
    if unknown:
        raise ValueError(f"Unknown selected features: {unknown}")
    forbidden = [column for column in selected if column in excluded]
    if forbidden:
        raise ValueError(f"Selected features are excluded by policy: {forbidden}")
    return selected


def _parse_class_ratio(value: str | None, split_name: str) -> dict[str, int] | None:
    if value in (None, "", False):
        return None
    match = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", str(value))
    if not match:
        raise ValueError(
            f"Invalid {split_name} class ratio {value!r}; expected normal:attack such as '8:2'."
        )
    normal, attack = (int(item) for item in match.groups())
    if normal <= 0 or attack <= 0:
        raise ValueError(f"{split_name} class ratio values must both be greater than zero.")
    return {"normal": normal, "attack": attack}


def configured_class_ratios(config: dict) -> dict[str, dict[str, int] | None]:
    configured = config.get("class_ratio")
    if isinstance(configured, dict):
        train_value = configured.get("train")
        test_value = configured.get("test")
    else:
        train_value = test_value = configured
    return {
        "train": _parse_class_ratio(train_value, "train"),
        "test": _parse_class_ratio(test_value, "test"),
    }


def configured_thresholds(config: dict) -> list[float]:
    values = config.get("decision_thresholds") or [0.5]
    thresholds = [float(value) for value in values]
    invalid = [value for value in thresholds if value <= 0 or value >= 1]
    if invalid:
        raise ValueError(f"Decision thresholds must be between 0 and 1: {invalid}")
    return thresholds


def _normalize_class_weight(value):
    if isinstance(value, str):
        if value.lower() in {"none", "null"}:
            return None
        return value
    if isinstance(value, dict):
        return {int(key): float(weight) for key, weight in value.items()}
    if value is None:
        return None
    raise ValueError(f"Unsupported class_weight value: {value!r}")


def configured_class_weight_grid(config: dict) -> list:
    values = config.get("class_weight_grid")
    if not values:
        values = config.get("parameter_distributions", {}).get("class_weight") or [None]
    return [_normalize_class_weight(value) for value in values]


def _class_weight_name(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, str):
        return value
    return "_".join(f"{key}-{value[key]:g}" for key in sorted(value))


def _class_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame[LABEL_COLUMN].astype(int).value_counts()
    return {"0": int(counts.get(0, 0)), "1": int(counts.get(1, 0))}


def sample_class_ratio(
    frame: pd.DataFrame,
    ratio: dict[str, int] | None,
    *,
    split_name: str,
    random_seed: int,
) -> tuple[pd.DataFrame, dict]:
    before = _class_counts(frame)
    audit = {
        "configured_ratio_normal_attack": ratio,
        "class_distribution_before": before,
        "class_distribution_after": before,
        "sampling_method": "none",
    }
    if ratio is None:
        return frame, audit
    divisor = math.gcd(ratio["normal"], ratio["attack"])
    normal_units = ratio["normal"] // divisor
    attack_units = ratio["attack"] // divisor
    multiplier = min(before["0"] // normal_units, before["1"] // attack_units)
    if multiplier < 1:
        raise ValueError(
            f"Cannot apply {split_name} class ratio {ratio}: labels={before}."
        )
    targets = {
        0: normal_units * multiplier,
        1: attack_units * multiplier,
    }
    sampled_parts = [
        frame[frame[LABEL_COLUMN].astype(int) == label].sample(
            n=target, random_state=random_seed + label
        )
        for label, target in targets.items()
    ]
    sampled = pd.concat(sampled_parts, ignore_index=True).sample(
        frac=1, random_state=random_seed
    ).reset_index(drop=True)
    audit["class_distribution_after"] = _class_counts(sampled)
    audit["sampling_method"] = "deterministic_downsample_excess_class"
    return sampled, audit


def _metric_row(y_true, y_pred, probabilities, *, threshold: float, class_weight, model_name: str) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "model_name": model_name,
        "class_weight": json.dumps(class_weight, sort_keys=True) if isinstance(class_weight, dict) else class_weight,
        "threshold": threshold,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "predicted_attack_count": int(y_pred.sum()),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def _run_paths(root: Path, run_version: str) -> dict[str, Path]:
    run_dir = run_bundle_dir(root, run_version)
    return {
        "run_dir": run_dir,
        "model": run_dir / f"randomforest_{run_version}.joblib",
        "metrics": run_dir / f"metrics_run_{run_version}.json",
        "manifest": run_dir / f"run_manifest_{run_version}.yaml",
        "notebook": run_dir / f"detection_run_{run_version}.executed.ipynb",
        "confusion_matrix": run_dir / f"confusion_matrix_run_{run_version}.csv",
        "importance": run_dir / f"feature_importance_run_{run_version}.csv",
        "summary": root / "metrics" / "metrics_summary.csv",
    }


def _scratch_paths(root: Path, scratch_name: str) -> dict[str, Path]:
    if not SCRATCH_NAME_PATTERN.fullmatch(scratch_name):
        raise ValueError(
            "Invalid scratch name; use 1-100 letters, numbers, dots, dashes, or underscores."
        )
    scratch_dir = root / "scratch" / scratch_name
    return {
        "run_dir": scratch_dir,
        "model": scratch_dir / "randomforest_scratch.joblib",
        "metrics": scratch_dir / "metrics_scratch.json",
        "manifest": scratch_dir / "scratch_manifest.yaml",
        "confusion_matrix": scratch_dir / "confusion_matrix_scratch.csv",
        "importance": scratch_dir / "feature_importance_scratch.csv",
    }


def _resolve_dataset_csv(root: Path, dataset_version: str) -> Path:
    path = root / "data" / "processed" / f"features_{dataset_version}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    return path


def _normalize_dataset_versions(dataset_versions: list[str] | str) -> list[str]:
    if isinstance(dataset_versions, str):
        values = [item for item in re.split(r"[\s,]+", dataset_versions.strip()) if item]
    else:
        values = [str(item).strip() for item in dataset_versions if str(item).strip()]
    if not values:
        raise ValueError("At least one dataset version is required.")
    return [dataset_version_from_features(f"features_{normalize_run_version(value)}.csv") for value in values]


def build_run_input_manifest(
    *,
    dataset_versions: list[str] | str | None = None,
    train_dataset_versions: list[str] | str | None = None,
    test_dataset_versions: list[str] | str | None = None,
    run_version: str,
    project_root: str | Path = ".",
    config_path: str | Path = "config/randomforest.yaml",
    overwrite: bool = False,
) -> dict:
    root = Path(project_root).resolve()
    run_version = normalize_run_version(run_version)
    if train_dataset_versions is not None or test_dataset_versions is not None:
        if train_dataset_versions is None or test_dataset_versions is None:
            raise ValueError("Scenario split requires both train_dataset_versions and test_dataset_versions.")
        train_dataset_versions = _normalize_dataset_versions(train_dataset_versions)
        test_dataset_versions = _normalize_dataset_versions(test_dataset_versions)
        overlap = sorted(set(train_dataset_versions) & set(test_dataset_versions))
        if overlap:
            raise ValueError(f"Scenario split datasets must not overlap: {overlap}")
        dataset_versions = list(dict.fromkeys(train_dataset_versions + test_dataset_versions))
        evaluation_strategy = "scenario_split_by_dataset_version"
    else:
        if dataset_versions is None:
            raise ValueError("At least one dataset version is required.")
        dataset_versions = _normalize_dataset_versions(dataset_versions)
        train_dataset_versions = None
        test_dataset_versions = None
        evaluation_strategy = "stratified_random_row_split_across_selected_datasets"
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    run_dir = run_bundle_dir(root, run_version)
    manifest_path = run_dir / f"run_input_manifest_{run_version}.yaml"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Run input manifest already exists: {manifest_path}")
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = [_resolve_dataset_csv(root, version) for version in dataset_versions]
    manifest = {
        "run_version": run_version,
        "dataset_versions": dataset_versions,
        "train_dataset_versions": train_dataset_versions,
        "test_dataset_versions": test_dataset_versions,
        "evaluation_strategy": evaluation_strategy,
        "source_processed_csv_files": [str(path) for path in csv_paths],
        "source_processed_csv_checksums": {str(path): file_checksum(path) for path in csv_paths},
        "config_path": str(config_file),
        "config_checksum": file_checksum(config_file),
        "feature_schema_checksum": schema_checksum(),
        "selected_feature_columns": selected_feature_columns(load_config(config_file)),
        "excluded_feature_columns": excluded_feature_columns(load_config(config_file)),
        "class_ratio": configured_class_ratios(load_config(config_file)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False, allow_unicode=False)
    return {"manifest": manifest, "manifest_path": manifest_path, "dataset_versions": dataset_versions}


def _load_frames(root: Path, dataset_versions: list[str]) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    csv_paths: list[Path] = []
    for dataset_version in dataset_versions:
        csv_path = _resolve_dataset_csv(root, dataset_version)
        csv_paths.append(csv_path)
        frame = pd.read_csv(csv_path, usecols=FEATURE_COLUMNS + [LABEL_COLUMN])
        frame["source_dataset_version"] = dataset_version
        frame["source_processed_csv"] = str(csv_path)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    return combined, csv_paths


def _require_training_data(frame: pd.DataFrame, dataset_versions: list[str]) -> None:
    missing = [column for column in FEATURE_COLUMNS + [LABEL_COLUMN] if column not in frame.columns]
    if missing:
        raise ValueError(f"Processed CSV is missing required columns: {missing}")
    counts = frame[LABEL_COLUMN].astype(int).value_counts()
    if len(counts) != 2:
        raise ValueError(
            "Training requires both normal and attack rows across the selected datasets. "
            f"Dataset versions={dataset_versions}, labels={counts.to_dict()}"
        )
    if int(counts.min()) < 4:
        raise ValueError(
            "Training requires at least four rows for each label to split and tune reliably. "
            f"Dataset versions={dataset_versions}, labels={counts.to_dict()}"
        )


def _require_scenario_test_data(frame: pd.DataFrame, dataset_versions: list[str]) -> None:
    missing = [column for column in FEATURE_COLUMNS + [LABEL_COLUMN] if column not in frame.columns]
    if missing:
        raise ValueError(f"Processed CSV is missing required columns: {missing}")
    counts = frame[LABEL_COLUMN].astype(int).value_counts()
    if len(counts) != 2:
        raise ValueError(
            "Scenario test datasets must contain both normal and attack rows for evaluation. "
            f"Dataset versions={dataset_versions}, labels={counts.to_dict()}"
        )


def _write_summary(summary_path: Path, metrics: dict, overwrite: bool) -> None:
    scalar_fields = [
        "run_version",
        "dataset_versions",
        "evaluation_strategy",
        "class_ratio",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "false_positive_rate",
        "false_negative_rate",
        "roc_auc",
        "train_row_count",
        "test_row_count",
        "selected_params",
        "comment",
    ]
    rows: list[dict] = []
    existing_comment = ""
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        existing = [row for row in rows if row.get("run_version") == metrics["run_version"]]
        if existing and not overwrite:
            raise FileExistsError(f"Metrics summary already contains {metrics['run_version']}.")
        if existing:
            existing_comment = existing[0].get("comment", "")
        rows = [row for row in rows if row.get("run_version") != metrics["run_version"]]
    summary_row = {
        field: json.dumps(metrics[field], sort_keys=True)
        if field in {"dataset_versions", "selected_params", "class_ratio"}
        else metrics.get(field)
        for field in scalar_fields
    }
    summary_row["comment"] = existing_comment
    rows.append(summary_row)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields)
        writer.writeheader()
        writer.writerows(rows)


def train_model(
    *,
    dataset_versions: list[str] | str | None = None,
    train_dataset_versions: list[str] | str | None = None,
    test_dataset_versions: list[str] | str | None = None,
    run_version: str | None,
    run_manifest_path: str | Path | None = None,
    project_root: str | Path = ".",
    config_path: str | Path = "config/randomforest.yaml",
    overwrite: bool = False,
    output_mode: str = "official",
    scratch_name: str | None = None,
    feature_columns_override: list[str] | None = None,
) -> dict:
    root = Path(project_root).resolve()
    if output_mode not in {"official", "scratch"}:
        raise ValueError(f"Unknown output mode: {output_mode}")
    if output_mode == "official":
        if run_version is None:
            raise ValueError("Official training requires a run_version.")
        run_version = normalize_run_version(run_version)
        paths = _run_paths(root, run_version)
    else:
        if run_manifest_path is not None:
            raise ValueError("Scratch training accepts dataset versions directly, not an official run manifest.")
        scratch_name = scratch_name or datetime.now(timezone.utc).strftime("experiment_%Y%m%d_%H%M%S")
        paths = _scratch_paths(root, scratch_name)
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = load_config(config_file)
    if not overwrite:
        for name, path in paths.items():
            if name == "run_dir" or (output_mode == "official" and name == "summary"):
                continue
            if path.exists():
                raise FileExistsError(f"Training output already exists: {path}")
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    run_manifest = None
    if run_manifest_path is not None:
        run_manifest_file = Path(run_manifest_path)
        if not run_manifest_file.is_absolute():
            run_manifest_file = root / run_manifest_file
        run_manifest = yaml.safe_load(run_manifest_file.read_text(encoding="utf-8"))
        dataset_versions = run_manifest.get("dataset_versions", [])
        train_dataset_versions = run_manifest.get("train_dataset_versions")
        test_dataset_versions = run_manifest.get("test_dataset_versions")
        config_file = Path(run_manifest.get("config_path", config_file))
        if not config_file.is_absolute():
            config_file = root / config_file
        config = load_config(config_file)
    elif train_dataset_versions is not None or test_dataset_versions is not None:
        if train_dataset_versions is None or test_dataset_versions is None:
            raise ValueError("Scenario split requires both train_dataset_versions and test_dataset_versions.")
        train_dataset_versions = _normalize_dataset_versions(train_dataset_versions)
        test_dataset_versions = _normalize_dataset_versions(test_dataset_versions)
        overlap = sorted(set(train_dataset_versions) & set(test_dataset_versions))
        if overlap:
            raise ValueError(f"Scenario split datasets must not overlap: {overlap}")
        dataset_versions = list(dict.fromkeys(train_dataset_versions + test_dataset_versions))
    elif dataset_versions is None:
        raise ValueError("Provide dataset_versions, scenario split datasets, or run_manifest_path.")
    else:
        dataset_versions = _normalize_dataset_versions(dataset_versions)

    if feature_columns_override is not None:
        if output_mode != "scratch":
            raise ValueError("Feature override is available only for scratch training.")
        config = dict(config)
        config["selected_feature_columns"] = list(feature_columns_override)

    feature_columns = selected_feature_columns(config)
    random_seed = int(config["random_seed"])
    class_ratios = configured_class_ratios(config)
    if train_dataset_versions is not None and test_dataset_versions is not None:
        train_frame, train_csv_paths = _load_frames(root, train_dataset_versions)
        test_frame, test_csv_paths = _load_frames(root, test_dataset_versions)
        _require_training_data(train_frame, train_dataset_versions)
        _require_scenario_test_data(test_frame, test_dataset_versions)
        csv_paths = list(dict.fromkeys(train_csv_paths + test_csv_paths))
        train_frame, train_ratio_audit = sample_class_ratio(
            train_frame, class_ratios["train"], split_name="train", random_seed=random_seed
        )
        test_frame, test_ratio_audit = sample_class_ratio(
            test_frame, class_ratios["test"], split_name="test", random_seed=random_seed + 100
        )
        x_train = train_frame[feature_columns].apply(pd.to_numeric, errors="coerce").astype("float32")
        y_train = train_frame[LABEL_COLUMN].astype("int8")
        x_test = test_frame[feature_columns].apply(pd.to_numeric, errors="coerce").astype("float32")
        y_test = test_frame[LABEL_COLUMN].astype("int8")
        evaluation_strategy = "scenario_split_by_dataset_version"
    else:
        frame, csv_paths = _load_frames(root, dataset_versions)
        _require_training_data(frame, dataset_versions)
        train_frame, test_frame = train_test_split(
            frame,
            test_size=float(config["test_size"]),
            random_state=random_seed,
            stratify=frame[LABEL_COLUMN].astype("int8"),
        )
        train_frame, train_ratio_audit = sample_class_ratio(
            train_frame, class_ratios["train"], split_name="train", random_seed=random_seed
        )
        test_frame, test_ratio_audit = sample_class_ratio(
            test_frame, class_ratios["test"], split_name="test", random_seed=random_seed + 100
        )
        x_train = train_frame[feature_columns].apply(pd.to_numeric, errors="coerce").astype("float32")
        y_train = train_frame[LABEL_COLUMN].astype("int8")
        x_test = test_frame[feature_columns].apply(pd.to_numeric, errors="coerce").astype("float32")
        y_test = test_frame[LABEL_COLUMN].astype("int8")
        evaluation_strategy = "stratified_random_row_split_across_selected_datasets"
    minimum_train_class = int(y_train.value_counts().min())
    cv_folds = min(int(config["cv_folds"]), minimum_train_class)
    if cv_folds < 2:
        raise ValueError(f"Not enough training rows per class for cross validation: {y_train.value_counts().to_dict()}")

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    random_state=random_seed,
                    n_jobs=int(config["model_n_jobs"]),
                ),
            ),
        ]
    )
    params = {f"model__{key}": values for key, values in config["parameter_distributions"].items()}
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=params,
        n_iter=int(config["search_iterations"]),
        scoring=str(config["scoring"]),
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed),
        random_state=random_seed,
        n_jobs=int(config["search_n_jobs"]),
        pre_dispatch=int(config["search_n_jobs"]),
        refit=True,
        verbose=1,
    )
    search.fit(x_train, y_train)
    predicted = search.predict(x_test)
    probabilities = search.predict_proba(x_test)[:, 1] if hasattr(search, "predict_proba") else None
    tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    roc_auc = float(roc_auc_score(y_test, probabilities)) if probabilities is not None else None
    timestamp = datetime.now(timezone.utc).isoformat()
    metrics = {
        "run_version": run_version,
        "output_mode": output_mode,
        "scratch_name": scratch_name if output_mode == "scratch" else None,
        "dataset_versions": dataset_versions,
        "train_dataset_versions": train_dataset_versions,
        "test_dataset_versions": test_dataset_versions,
        "source_processed_csv_files": [str(path) for path in csv_paths],
        "feature_schema_checksum": schema_checksum(),
        "feature_columns": feature_columns,
        "excluded_feature_columns": excluded_feature_columns(config),
        "random_seed": random_seed,
        "class_ratio": class_ratios,
        "class_ratio_sampling_train": train_ratio_audit,
        "class_ratio_sampling_test": test_ratio_audit,
        "evaluation_strategy": evaluation_strategy,
        "hyperparameters_searched": params,
        "selected_params": search.best_params_,
        "model_output_filename": str(paths["model"]),
        "execution_timestamp": timestamp,
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "f1_score": float(f1_score(y_test, predicted, zero_division=0)),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "roc_auc": roc_auc,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "train_row_count": int(len(x_train)),
        "test_row_count": int(len(x_test)),
        "class_distribution_train": {str(key): int(value) for key, value in y_train.value_counts().items()},
        "class_distribution_test": {str(key): int(value) for key, value in y_test.value_counts().items()},
    }
    artifact = {
        "model": search.best_estimator_,
        "feature_columns": feature_columns,
        "best_params": search.best_params_,
        "run_version": run_version,
        "output_mode": output_mode,
        "scratch_name": scratch_name if output_mode == "scratch" else None,
        "dataset_versions": dataset_versions,
        "train_dataset_versions": train_dataset_versions,
        "test_dataset_versions": test_dataset_versions,
        "excluded_feature_columns": excluded_feature_columns(config),
        "class_ratio": class_ratios,
        "metrics": metrics,
        "feature_schema_checksum": metrics["feature_schema_checksum"],
        "training_timestamp": timestamp,
    }
    joblib.dump(artifact, paths["model"])
    with paths["metrics"].open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with paths["confusion_matrix"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "predicted_normal", "predicted_attack"])
        writer.writerow(["actual_normal", int(tn), int(fp)])
        writer.writerow(["actual_attack", int(fn), int(tp)])
    importances = search.best_estimator_.named_steps["model"].feature_importances_
    pd.DataFrame({"feature": feature_columns, "importance": importances}).sort_values(
        "importance", ascending=False
    ).to_csv(paths["importance"], index=False)
    run_manifest = {
        "run_version": run_version,
        "output_mode": output_mode,
        "scratch_name": scratch_name if output_mode == "scratch" else None,
        "dataset_versions": dataset_versions,
        "train_dataset_versions": train_dataset_versions,
        "test_dataset_versions": test_dataset_versions,
        "source_processed_csv_files": [str(path) for path in csv_paths],
        "run_dir": str(paths["run_dir"]),
        "model_output_filename": str(paths["model"]),
        "metrics_filename": str(paths["metrics"]),
        "executed_notebook_filename": str(paths["notebook"]) if "notebook" in paths else None,
        "feature_importance_filename": str(paths["importance"]),
        "confusion_matrix_filename": str(paths["confusion_matrix"]),
        "random_seed": random_seed,
        "evaluation_strategy": metrics["evaluation_strategy"],
        "hyperparameters_searched": params,
        "selected_params": search.best_params_,
        "class_ratio": class_ratios,
        "class_ratio_sampling_train": metrics["class_ratio_sampling_train"],
        "class_ratio_sampling_test": metrics["class_ratio_sampling_test"],
        "feature_schema_checksum": metrics["feature_schema_checksum"],
        "excluded_feature_columns": metrics["excluded_feature_columns"],
        "execution_timestamp": timestamp,
        "class_distribution_train": metrics["class_distribution_train"],
        "class_distribution_test": metrics["class_distribution_test"],
    }
    if run_manifest_path is not None and run_manifest:
        run_manifest.update(
            {
                "input_manifest_path": str(Path(run_manifest_path).resolve()),
                "input_manifest_checksum": file_checksum(run_manifest_path),
            }
        )
    with paths["manifest"].open("w", encoding="utf-8") as handle:
        yaml.safe_dump(run_manifest, handle, sort_keys=False, allow_unicode=False)
    if output_mode == "official":
        _write_summary(paths["summary"], metrics, overwrite)
    return {"metrics": metrics, "paths": paths, "manifest": run_manifest}


def train_scratch_model(
    *,
    dataset_versions: list[str] | str | None = None,
    train_dataset_versions: list[str] | str | None = None,
    test_dataset_versions: list[str] | str | None = None,
    scratch_name: str | None = None,
    project_root: str | Path = ".",
    config_path: str | Path = "config/randomforest.yaml",
    feature_columns: list[str] | None = None,
    overwrite: bool = False,
) -> dict:
    return train_model(
        dataset_versions=dataset_versions,
        train_dataset_versions=train_dataset_versions,
        test_dataset_versions=test_dataset_versions,
        run_version=None,
        project_root=project_root,
        config_path=config_path,
        overwrite=overwrite,
        output_mode="scratch",
        scratch_name=scratch_name,
        feature_columns_override=feature_columns,
    )


def load_official_model(run_version: str, project_root: str | Path = ".") -> dict:
    root = Path(project_root).resolve()
    normalized = normalize_run_version(run_version)
    model_path = _run_paths(root, normalized)["model"]
    if not model_path.exists():
        raise FileNotFoundError(f"Official trained model not found: {model_path}")
    return joblib.load(model_path)


def _rebuild_evaluation_frames_from_metrics(root: Path, metrics: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    random_seed = int(metrics["random_seed"])
    class_ratio = metrics.get("class_ratio") or {"train": None, "test": None}
    if metrics.get("train_dataset_versions") and metrics.get("test_dataset_versions"):
        train_frame, train_csv_paths = _load_frames(root, metrics["train_dataset_versions"])
        test_frame, test_csv_paths = _load_frames(root, metrics["test_dataset_versions"])
        train_frame, _ = sample_class_ratio(
            train_frame, class_ratio.get("train"), split_name="train", random_seed=random_seed
        )
        test_frame, _ = sample_class_ratio(
            test_frame, class_ratio.get("test"), split_name="test", random_seed=random_seed + 100
        )
        return train_frame, test_frame, list(dict.fromkeys(train_csv_paths + test_csv_paths))

    frame, csv_paths = _load_frames(root, metrics["dataset_versions"])
    train_frame, test_frame = train_test_split(
        frame,
        test_size=float(metrics.get("test_size", 0.2)),
        random_state=random_seed,
        stratify=frame[LABEL_COLUMN].astype("int8"),
    )
    train_frame, _ = sample_class_ratio(
        train_frame, class_ratio.get("train"), split_name="train", random_seed=random_seed
    )
    test_frame, _ = sample_class_ratio(
        test_frame, class_ratio.get("test"), split_name="test", random_seed=random_seed + 100
    )
    return train_frame, test_frame, csv_paths


def compare_threshold_weight_grid(
    *,
    run_version: str,
    project_root: str | Path = ".",
    config_path: str | Path = "config/randomforest.yaml",
    analysis_name: str | None = None,
    overwrite: bool = False,
) -> dict:
    root = Path(project_root).resolve()
    normalized = normalize_run_version(run_version)
    paths = _run_paths(root, normalized)
    artifact = load_official_model(normalized, root)
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = load_config(config_file)
    thresholds = configured_thresholds(config)
    class_weight_grid = configured_class_weight_grid(config)
    analysis_name = analysis_name or f"threshold_weight_{normalized}"
    if not SCRATCH_NAME_PATTERN.fullmatch(analysis_name):
        raise ValueError("Invalid analysis name; use 1-100 letters, numbers, dots, dashes, or underscores.")
    output_dir = root / "scratch" / analysis_name
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Analysis output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_frame, test_frame, csv_paths = _rebuild_evaluation_frames_from_metrics(root, metrics)
    feature_columns = artifact["feature_columns"]
    x_train = train_frame[feature_columns].apply(pd.to_numeric, errors="coerce").astype("float32")
    y_train = train_frame[LABEL_COLUMN].astype("int8")
    x_test = test_frame[feature_columns].apply(pd.to_numeric, errors="coerce").astype("float32")
    y_test = test_frame[LABEL_COLUMN].astype("int8")
    base_model = artifact["model"]
    base_probabilities = base_model.predict_proba(x_test)[:, 1]

    rows = []
    for threshold in thresholds:
        rows.append(
            _metric_row(
                y_test,
                (base_probabilities >= threshold).astype("int8"),
                base_probabilities,
                threshold=threshold,
                class_weight=artifact.get("best_params", {}).get("model__class_weight"),
                model_name="base_run_model",
            )
        )

    best_params = {
        key.replace("model__", ""): value
        for key, value in artifact.get("best_params", {}).items()
        if key.startswith("model__")
    }
    best_params.pop("class_weight", None)
    random_seed = int(metrics["random_seed"])
    for class_weight in class_weight_grid:
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        **best_params,
                        class_weight=class_weight,
                        random_state=random_seed,
                        n_jobs=int(config["model_n_jobs"]),
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        model_name = f"class_weight_{_class_weight_name(class_weight)}"
        for threshold in thresholds:
            rows.append(
                _metric_row(
                    y_test,
                    (probabilities >= threshold).astype("int8"),
                    probabilities,
                    threshold=threshold,
                    class_weight=class_weight,
                    model_name=model_name,
                )
            )

    comparison = pd.DataFrame(rows).sort_values(["model_name", "threshold"])
    csv_path = output_dir / "threshold_weight_confusion_matrix.csv"
    json_path = output_dir / "threshold_weight_manifest.json"
    comparison.to_csv(csv_path, index=False)
    manifest = {
        "analysis_name": analysis_name,
        "base_run_version": normalized,
        "base_model": str(paths["model"]),
        "source_processed_csv_files": [str(path) for path in csv_paths],
        "thresholds": thresholds,
        "class_weight_grid": class_weight_grid,
        "feature_columns": feature_columns,
        "train_row_count": int(len(x_train)),
        "test_row_count": int(len(x_test)),
        "class_distribution_train": _class_counts(train_frame),
        "class_distribution_test": _class_counts(test_frame),
        "comparison_csv": str(csv_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {"comparison": comparison, "csv_path": csv_path, "manifest_path": json_path, "output_dir": output_dir}


def notebook_training_entry() -> None:
    result = train_model(
        dataset_versions=os.environ.get("ML_DATASET_VERSIONS"),
        run_version=os.environ["ML_RUN_VERSION"],
        project_root=os.environ["ML_PROJECT_ROOT"],
        config_path=os.environ.get("ML_CONFIG", "config/randomforest.yaml"),
        run_manifest_path=os.environ.get("ML_RUN_MANIFEST"),
        overwrite=os.environ.get("ML_OVERWRITE", "").lower() in {"1", "true", "yes"},
    )
    metrics = result["metrics"]
    print("Random Forest Training Summary")
    print("==============================")
    print(f"Run version: {metrics['run_version']}")
    print(f"Dataset versions: {metrics['dataset_versions']}")
    print(f"Rows: train={metrics['train_row_count']} test={metrics['test_row_count']}")
    print(f"Labels train: {metrics['class_distribution_train']}")
    print(f"Labels test: {metrics['class_distribution_test']}")
    print(f"Selected parameters: {metrics['selected_params']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 score: {metrics['f1_score']:.4f}")
    print(f"False positive rate: {metrics['false_positive_rate']:.4f}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
    print(f"Saved model: {result['paths']['model']}")


def execute_training_notebook(
    *,
    dataset_versions: list[str] | str | None = None,
    run_version: str,
    run_manifest_path: str | Path | None = None,
    project_root: str | Path = ".",
    config_path: str | Path = "config/randomforest.yaml",
    overwrite: bool = False,
) -> Path:
    import nbformat

    root = Path(project_root).resolve()
    notebook_path = root / "notebook" / "detection.ipynb"
    run_dir = run_bundle_dir(root, run_version)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_path = run_dir / f"detection_run_{normalize_run_version(run_version)}.executed.ipynb"
    if run_path.exists() and not overwrite:
        raise FileExistsError(f"Executed notebook already exists: {run_path}")
    old_environment = {
        key: os.environ.get(key)
        for key in (
            "ML_DATASET_VERSIONS",
            "ML_DATASET_CSVS",
            "ML_RUN_MANIFEST",
            "ML_RUN_VERSION",
            "ML_PROJECT_ROOT",
            "ML_CONFIG",
            "ML_OVERWRITE",
        )
    }
    if run_manifest_path is not None:
        run_manifest_file = Path(run_manifest_path)
        if not run_manifest_file.is_absolute():
            run_manifest_file = root / run_manifest_file
        run_manifest_data = yaml.safe_load(run_manifest_file.read_text(encoding="utf-8"))
        normalized_versions = _normalize_dataset_versions(run_manifest_data["dataset_versions"])
        dataset_csvs = [str(_resolve_dataset_csv(root, version)) for version in normalized_versions]
        os.environ["ML_RUN_MANIFEST"] = str(run_manifest_file)
    else:
        if dataset_versions is None:
            raise ValueError("Provide either dataset_versions or run_manifest_path.")
        normalized_versions = _normalize_dataset_versions(dataset_versions)
        dataset_csvs = [str(_resolve_dataset_csv(root, version)) for version in normalized_versions]
    os.environ.update(
        {
            "ML_DATASET_VERSIONS": " ".join(normalized_versions),
            "ML_DATASET_CSVS": os.pathsep.join(dataset_csvs),
            "ML_RUN_VERSION": normalize_run_version(run_version),
            "ML_PROJECT_ROOT": str(root),
            "ML_CONFIG": str(config_path),
            "ML_OVERWRITE": str(overwrite),
        }
    )
    try:
        notebook = nbformat.read(notebook_path, as_version=4)
        namespace: dict = {"__name__": "__main__"}
        execution_count = 0
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            execution_count += 1
            cell.execution_count = execution_count
            cell.outputs = []
            output = io.StringIO()
            try:
                with redirect_stdout(output), redirect_stderr(output):
                    exec(compile(cell.source, f"{notebook_path}#cell-{execution_count}", "exec"), namespace)
            except Exception as exc:
                text = output.getvalue()
                if text:
                    cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=text))
                cell.outputs.append(
                    nbformat.v4.new_output(
                        "error",
                        ename=type(exc).__name__,
                        evalue=str(exc),
                        traceback=traceback.format_exc().splitlines(),
                    )
                )
                nbformat.write(notebook, run_path)
                raise
            text = output.getvalue()
            if text:
                cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=text))
        nbformat.write(notebook, run_path)
    finally:
        for key, previous in old_environment.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
    return run_path
