from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from .schema import FEATURE_COLUMNS, LABEL_COLUMN
from .training import (
    _class_counts,
    _rebuild_evaluation_frames_from_metrics,
    _run_paths,
    load_config,
    load_official_model,
)
from .versioning import normalize_run_version


FIXED_THRESHOLD = 0.3

DROPPED_SHORTCUT_FEATURES = [
    "frame_len",
    "ip_len",
    "frame_cap_len",
    "udp_length",
    "src_port",
    "query_name_len",
    "query_entropy",
    "query_label_count",
    "ttl_mean",
    "ttl_min",
    "ttl_max",
    "ttl_std",
]

TOP_IMPORTANCE_FEATURES = [
    "has_additional_A",
    "additional_A_count",
    "unique_record_name_count",
    "additional_out_of_bailiwick_count",
    "record_total",
    "answer_A_count",
    "dns_count_add_rr",
    "dns_count_answers",
    "additional_AAAA_count",
    "answer_record_count",
]


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return cleaned.strip("._-") or "unnamed"


def _in_schema(columns: list[str]) -> list[str]:
    return [column for column in columns if column in FEATURE_COLUMNS]


def _dns_structure_features() -> list[str]:
    prefixes = (
        "dns_flags_",
        "dns_count_",
        "answer_",
        "authority_",
        "additional_",
        "has_",
    )
    exact = {
        "record_total",
        "unique_record_name_count",
        "answer_matches_query_count",
        "additional_out_of_bailiwick_count",
    }
    return [
        column
        for column in FEATURE_COLUMNS
        if column.startswith(prefixes) or column in exact
    ]


def build_feature_sets(base_features: list[str]) -> dict[str, list[str]]:
    base = [column for column in base_features if column in FEATURE_COLUMNS]
    dropped = _in_schema(DROPPED_SHORTCUT_FEATURES)
    top = _in_schema(TOP_IMPORTANCE_FEATURES)
    dns_structure = _dns_structure_features()
    shortcut_suspects = _in_schema(list(dict.fromkeys(dropped + top)))

    feature_sets: dict[str, list[str]] = {
        "all_current": base,
        "dropped_shortcut_only": dropped,
        "top_importance_only": top,
        "dns_structure_only": dns_structure,
        "shortcut_suspects_only": shortcut_suspects,
    }
    for feature in top:
        if feature in base and len(base) > 1:
            feature_sets[f"ablate_top__{feature}"] = [
                column for column in base if column != feature
            ]
    for feature in dropped:
        if feature not in base:
            feature_sets[f"add_back_dropped__{feature}"] = base + [feature]
    return {
        name: list(dict.fromkeys(columns))
        for name, columns in feature_sets.items()
        if columns
    }


def _base_model_params(artifact: dict, config: dict, random_seed: int) -> dict:
    params = {
        key.replace("model__", ""): value
        for key, value in artifact.get("best_params", {}).items()
        if key.startswith("model__")
    }
    params.setdefault("random_state", random_seed)
    params["random_state"] = random_seed
    params["n_jobs"] = int(config.get("model_n_jobs", 1))
    return params


def _metric_row(
    *,
    feature_set: str,
    feature_count: int,
    y_true,
    y_pred,
    probabilities,
) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "feature_set": feature_set,
        "feature_count": int(feature_count),
        "threshold": FIXED_THRESHOLD,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "attack_probability_mean": float(pd.Series(probabilities).mean()),
        "attack_probability_median": float(pd.Series(probabilities).median()),
    }


def _probability_stats(values) -> dict:
    series = pd.Series(values, dtype="float64").dropna()
    if series.empty:
        return {
            "count": 0,
            "mean": math.nan,
            "median": math.nan,
            "std": math.nan,
            "min": math.nan,
            "max": math.nan,
            "q25": math.nan,
            "q75": math.nan,
            "IQR": math.nan,
        }
    q25 = series.quantile(0.25)
    q75 = series.quantile(0.75)
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
        "min": float(series.min()),
        "max": float(series.max()),
        "q25": float(q25),
        "q75": float(q75),
        "IQR": float(q75 - q25),
    }


def _probability_rows(feature_set: str, y_true, y_pred, probabilities) -> list[dict]:
    y_true = pd.Series(y_true).astype(int).reset_index(drop=True)
    y_pred = pd.Series(y_pred).astype(int).reset_index(drop=True)
    probs = pd.Series(probabilities, dtype="float64").reset_index(drop=True)
    masks = {
        "actual_normal": y_true == 0,
        "actual_attack": y_true == 1,
        "TP": (y_true == 1) & (y_pred == 1),
        "FN": (y_true == 1) & (y_pred == 0),
        "FP": (y_true == 0) & (y_pred == 1),
        "TN": (y_true == 0) & (y_pred == 0),
    }
    rows = []
    for group, mask in masks.items():
        rows.append(
            {
                "feature_set": feature_set,
                "group": group,
                **_probability_stats(probs[mask]),
            }
        )
    return rows


def _series_stats(series: pd.Series) -> dict:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    missing = int(numeric.isna().sum())
    if valid.empty:
        return {
            "count": 0,
            "mean": math.nan,
            "median": math.nan,
            "std": math.nan,
            "min": math.nan,
            "max": math.nan,
            "q25": math.nan,
            "q75": math.nan,
            "IQR": math.nan,
            "missing_count": missing,
            "zero_ratio": math.nan,
            "unique_count": 0,
        }
    q25 = valid.quantile(0.25)
    q75 = valid.quantile(0.75)
    return {
        "count": int(valid.count()),
        "mean": float(valid.mean()),
        "median": float(valid.median()),
        "std": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
        "min": float(valid.min()),
        "max": float(valid.max()),
        "q25": float(q25),
        "q75": float(q75),
        "IQR": float(q75 - q25),
        "missing_count": missing,
        "zero_ratio": float((valid == 0).mean()),
        "unique_count": int(valid.nunique(dropna=True)),
    }


def _cohens_d(left: pd.Series, right: pd.Series) -> float:
    left = pd.to_numeric(left, errors="coerce").dropna()
    right = pd.to_numeric(right, errors="coerce").dropna()
    if len(left) < 2 or len(right) < 2:
        return math.nan
    left_std = left.std(ddof=1)
    right_std = right.std(ddof=1)
    pooled = math.sqrt(
        ((len(left) - 1) * left_std**2 + (len(right) - 1) * right_std**2)
        / (len(left) + len(right) - 2)
    )
    if pooled == 0:
        return math.nan
    return float((left.mean() - right.mean()) / pooled)


def _mann_whitney_p(left: pd.Series, right: pd.Series) -> float:
    left = pd.to_numeric(left, errors="coerce").dropna()
    right = pd.to_numeric(right, errors="coerce").dropna()
    if left.empty or right.empty:
        return math.nan
    try:
        return float(mannwhitneyu(left, right, alternative="two-sided").pvalue)
    except ValueError:
        return math.nan


def _comparison_row(
    frame: pd.DataFrame,
    *,
    feature_set: str,
    feature: str,
    comparison: str,
    left_group: str,
    left_mask,
    right_group: str,
    right_mask,
) -> dict:
    left = frame.loc[left_mask, feature]
    right = frame.loc[right_mask, feature]
    left_stats = _series_stats(left)
    right_stats = _series_stats(right)
    denom = abs(left_stats["IQR"]) + abs(right_stats["IQR"]) + 1e-9
    if math.isnan(left_stats["median"]) or math.isnan(right_stats["median"]):
        separation = math.nan
    else:
        separation = float(abs(left_stats["median"] - right_stats["median"]) / denom)
    row = {
        "feature_set": feature_set,
        "feature": feature,
        "comparison": comparison,
        "left_group": left_group,
        "right_group": right_group,
        "mann_whitney_u_pvalue": _mann_whitney_p(left, right),
        "cohens_d": _cohens_d(left, right),
        "separation_score": separation,
    }
    for key, value in left_stats.items():
        row[f"left_{key}"] = value
    for key, value in right_stats.items():
        row[f"right_{key}"] = value
    return row


def _statistic_rows(
    *,
    feature_set: str,
    test_frame: pd.DataFrame,
    features: list[str],
    y_true,
    y_pred,
) -> list[dict]:
    frame = test_frame.reset_index(drop=True).copy()
    frame["_actual"] = pd.Series(y_true).astype(int).reset_index(drop=True)
    frame["_predicted"] = pd.Series(y_pred).astype(int).reset_index(drop=True)
    frame["_outcome"] = "TN"
    frame.loc[(frame["_actual"] == 1) & (frame["_predicted"] == 1), "_outcome"] = "TP"
    frame.loc[(frame["_actual"] == 1) & (frame["_predicted"] == 0), "_outcome"] = "FN"
    frame.loc[(frame["_actual"] == 0) & (frame["_predicted"] == 1), "_outcome"] = "FP"

    comparisons = [
        ("actual_normal_vs_actual_attack", "actual_normal", frame["_actual"] == 0, "actual_attack", frame["_actual"] == 1),
        ("TP_vs_FN", "TP", frame["_outcome"] == "TP", "FN", frame["_outcome"] == "FN"),
        ("TN_vs_FP", "TN", frame["_outcome"] == "TN", "FP", frame["_outcome"] == "FP"),
        ("TP_vs_TN", "TP", frame["_outcome"] == "TP", "TN", frame["_outcome"] == "TN"),
        ("FN_vs_TP", "FN", frame["_outcome"] == "FN", "TP", frame["_outcome"] == "TP"),
    ]
    rows: list[dict] = []
    for feature in features:
        for comparison, left_group, left_mask, right_group, right_mask in comparisons:
            rows.append(
                _comparison_row(
                    frame,
                    feature_set=feature_set,
                    feature=feature,
                    comparison=comparison,
                    left_group=left_group,
                    left_mask=left_mask,
                    right_group=right_group,
                    right_mask=right_mask,
                )
            )
    return rows


def _pyplot():
    cache_dir = Path(os.environ.get("MPLCONFIGDIR", Path(tempfile.gettempdir()) / "ml_teamproject_matplotlib"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_metrics(metrics: pd.DataFrame, plots_dir: Path) -> None:
    plt = _pyplot()
    columns = ["recall", "precision", "false_positive_rate", "f1"]
    plot_frame = metrics.set_index("feature_set")[columns]
    width = max(12, min(30, len(plot_frame) * 0.55))
    ax = plot_frame.plot(kind="bar", figsize=(width, 6))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Feature set metrics at threshold=0.3")
    ax.tick_params(axis="x", labelrotation=75)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(plots_dir / "metrics_comparison.png", dpi=160)
    plt.close()


def _plot_confusion(row: dict, plots_dir: Path) -> None:
    plt = _pyplot()
    matrix = [[row["TN"], row["FP"]], [row["FN"], row["TP"]]]
    fig, ax = plt.subplots(figsize=(4, 3.6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred_normal", "pred_attack"])
    ax.set_yticks([0, 1], labels=["actual_normal", "actual_attack"])
    ax.set_title(row["feature_set"])
    for y_index, values in enumerate(matrix):
        for x_index, value in enumerate(values):
            ax.text(x_index, y_index, str(value), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(plots_dir / f"confusion_matrix_{_safe_name(row['feature_set'])}.png", dpi=160)
    plt.close()


def _plot_importance(importances: pd.DataFrame, feature_set: str, plots_dir: Path) -> None:
    plt = _pyplot()
    subset = importances[importances["feature_set"] == feature_set].head(20)
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(9, max(4, len(subset) * 0.28)))
    ax.barh(subset["feature"][::-1], subset["importance"][::-1])
    ax.set_xlabel("importance")
    ax.set_title(f"Top feature importances: {feature_set}")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(plots_dir / f"feature_importance_{_safe_name(feature_set)}.png", dpi=160)
    plt.close()


def _plot_feature_distributions(
    frame: pd.DataFrame,
    *,
    feature_set: str,
    features: list[str],
    y_true,
    y_pred,
    plots_dir: Path,
) -> None:
    plt = _pyplot()
    plot_frame = frame.reset_index(drop=True).copy()
    plot_frame["_actual"] = pd.Series(y_true).astype(int).reset_index(drop=True)
    plot_frame["_predicted"] = pd.Series(y_pred).astype(int).reset_index(drop=True)
    plot_frame["_outcome"] = "TN"
    plot_frame.loc[(plot_frame["_actual"] == 1) & (plot_frame["_predicted"] == 1), "_outcome"] = "TP"
    plot_frame.loc[(plot_frame["_actual"] == 1) & (plot_frame["_predicted"] == 0), "_outcome"] = "FN"
    plot_frame.loc[(plot_frame["_actual"] == 0) & (plot_frame["_predicted"] == 1), "_outcome"] = "FP"
    for feature in features:
        if feature not in plot_frame:
            continue
        normal = pd.to_numeric(plot_frame.loc[plot_frame["_actual"] == 0, feature], errors="coerce").dropna()
        attack = pd.to_numeric(plot_frame.loc[plot_frame["_actual"] == 1, feature], errors="coerce").dropna()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(normal, bins=30, alpha=0.55, label="actual_normal")
        ax.hist(attack, bins=30, alpha=0.55, label="actual_attack")
        ax.set_title(f"{feature_set}: {feature} by actual label")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(plots_dir / f"hist_{_safe_name(feature_set)}_{_safe_name(feature)}_actual.png", dpi=150)
        plt.close()

        tp = pd.to_numeric(plot_frame.loc[plot_frame["_outcome"] == "TP", feature], errors="coerce").dropna()
        fn = pd.to_numeric(plot_frame.loc[plot_frame["_outcome"] == "FN", feature], errors="coerce").dropna()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(tp, bins=30, alpha=0.55, label="TP")
        ax.hist(fn, bins=30, alpha=0.55, label="FN")
        ax.set_title(f"{feature_set}: {feature} TP vs FN")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(plots_dir / f"hist_{_safe_name(feature_set)}_{_safe_name(feature)}_TP_FN.png", dpi=150)
        plt.close()

        box_values = [
            pd.to_numeric(plot_frame.loc[plot_frame["_outcome"] == outcome, feature], errors="coerce").dropna()
            for outcome in ["TP", "FN", "FP", "TN"]
        ]
        if any(len(values) for values in box_values):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.boxplot(box_values, tick_labels=["TP", "FN", "FP", "TN"], showfliers=False)
            ax.set_title(f"{feature_set}: {feature} outcome boxplot")
            ax.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            plt.savefig(plots_dir / f"boxplot_{_safe_name(feature_set)}_{_safe_name(feature)}.png", dpi=150)
            plt.close()


def _plot_probability_distribution(
    feature_set: str,
    y_true,
    y_pred,
    probabilities,
    plots_dir: Path,
    *,
    filename: str,
) -> None:
    plt = _pyplot()
    y_true = pd.Series(y_true).astype(int).reset_index(drop=True)
    y_pred = pd.Series(y_pred).astype(int).reset_index(drop=True)
    probs = pd.Series(probabilities, dtype="float64").reset_index(drop=True)
    groups = {
        "normal": y_true == 0,
        "attack": y_true == 1,
        "TP": (y_true == 1) & (y_pred == 1),
        "FN": (y_true == 1) & (y_pred == 0),
        "FP": (y_true == 0) & (y_pred == 1),
        "TN": (y_true == 0) & (y_pred == 0),
    }
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for group, mask in groups.items():
        values = probs[mask]
        if not values.empty:
            ax.hist(values, bins=30, alpha=0.35, density=True, label=group)
    ax.axvline(FIXED_THRESHOLD, color="black", linestyle="--", label="threshold=0.3")
    ax.set_xlabel("attack probability")
    ax.set_ylabel("density")
    ax.set_title(f"Prediction probability distribution: {feature_set}")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(plots_dir / filename, dpi=160)
    plt.close()


def _write_yaml(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def _notebook_table_cell(title: str, frame: pd.DataFrame, *, max_rows: int | None = None):
    import nbformat

    shown = frame if max_rows is None else frame.head(max_rows)
    cell = nbformat.v4.new_code_cell(f"# {title}\n{title.replace(' ', '_').lower()}")
    cell.execution_count = 1
    cell.outputs = [
        nbformat.v4.new_output(
            "execute_result",
            data={
                "text/plain": shown.to_string(index=False),
                "text/html": shown.to_html(index=False),
            },
            execution_count=1,
        )
    ]
    return cell


def write_feature_experiment_notebook(output_dir: str | Path) -> Path:
    import nbformat

    output_dir = Path(output_dir)
    metrics = pd.read_csv(output_dir / "metrics_summary.csv")
    confusion = pd.read_csv(output_dir / "confusion_matrices.csv")
    importances = pd.read_csv(output_dir / "feature_importances.csv")
    statistics = pd.read_csv(output_dir / "feature_statistics.csv")
    probability = pd.read_csv(output_dir / "probability_summary.csv")
    config = yaml.safe_load((output_dir / "config.yaml").read_text(encoding="utf-8"))
    plots_dir = output_dir / "plots"

    metric_columns = [
        "feature_set",
        "feature_count",
        "TN",
        "FP",
        "FN",
        "TP",
        "precision",
        "recall",
        "f1",
        "f2",
        "false_positive_rate",
        "false_negative_rate",
    ]
    metric_view = metrics[metric_columns].sort_values(
        ["recall", "precision", "false_positive_rate"], ascending=[False, False, True]
    )
    baseline = metrics[metrics["feature_set"] == "all_current"]
    best_recall = metric_view.head(5)
    tp_fn_stats = statistics[statistics["comparison"].isin(["TP_vs_FN", "FN_vs_TP"])].sort_values(
        ["mann_whitney_u_pvalue", "separation_score"], ascending=[True, False]
    )
    tn_fp_stats = statistics[statistics["comparison"] == "TN_vs_FP"].sort_values(
        ["mann_whitney_u_pvalue", "separation_score"], ascending=[True, False]
    )
    top_importance = importances.groupby("feature_set").head(15)

    cells = [
        nbformat.v4.new_markdown_cell(
            "# Feature Experiment Report\n\n"
            f"- Base run: `{config.get('base_run_version')}`\n"
            f"- Threshold: `{config.get('threshold')}`\n"
            f"- Class weight: `{config.get('class_weight')}`\n"
            f"- Train datasets: `{config.get('train_dataset_versions')}`\n"
            f"- Test datasets: `{config.get('test_dataset_versions')}`\n"
            f"- Output dir: `{output_dir}`\n\n"
            "이 노트북은 이미 생성된 CSV/PNG를 보기 좋게 묶은 리포트입니다. "
            "셀을 다시 실행해도 공식 run 산출물은 수정하지 않습니다."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            f"OUTPUT_DIR = Path({str(output_dir)!r})\n"
            "metrics = pd.read_csv(OUTPUT_DIR / 'metrics_summary.csv')\n"
            "confusion = pd.read_csv(OUTPUT_DIR / 'confusion_matrices.csv')\n"
            "importances = pd.read_csv(OUTPUT_DIR / 'feature_importances.csv')\n"
            "statistics = pd.read_csv(OUTPUT_DIR / 'feature_statistics.csv')\n"
            "probability = pd.read_csv(OUTPUT_DIR / 'probability_summary.csv')\n"
            "plots = OUTPUT_DIR / 'plots'\n"
        ),
        nbformat.v4.new_markdown_cell("## 1. 전체 feature set 성능 요약"),
        _notebook_table_cell("metrics", metric_view),
        nbformat.v4.new_markdown_cell("## 2. 기준 all_current와 recall 상위 feature set"),
        _notebook_table_cell("baseline_all_current", baseline[metric_columns]),
        _notebook_table_cell("top_recall_feature_sets", best_recall),
        nbformat.v4.new_markdown_cell("## 3. Confusion matrix 수치"),
        _notebook_table_cell("confusion_matrices", confusion),
        nbformat.v4.new_markdown_cell("## 4. TP/FN 분리 통계 상위"),
        _notebook_table_cell("tp_fn_statistics", tp_fn_stats, max_rows=80),
        nbformat.v4.new_markdown_cell("## 5. TN/FP 분리 통계 상위"),
        _notebook_table_cell("tn_fp_statistics", tn_fp_stats, max_rows=80),
        nbformat.v4.new_markdown_cell("## 6. Probability 요약"),
        _notebook_table_cell("probability_summary", probability),
        nbformat.v4.new_markdown_cell("## 7. Feature importance 상위"),
        _notebook_table_cell("top_feature_importances", top_importance, max_rows=120),
        nbformat.v4.new_markdown_cell(
            "## 8. 주요 그래프\n\n"
            "아래 이미지는 파일로 저장된 PNG를 노트북 안에서 바로 보여줍니다."
        ),
    ]

    image_names = [
        "metrics_comparison.png",
        "probability_distribution.png",
        "confusion_matrix_all_current.png",
        "feature_importance_all_current.png",
        "confusion_matrix_dropped_shortcut_only.png",
        "feature_importance_dropped_shortcut_only.png",
        "confusion_matrix_shortcut_suspects_only.png",
        "feature_importance_shortcut_suspects_only.png",
    ]
    for image_name in image_names:
        if (plots_dir / image_name).exists():
            cells.append(nbformat.v4.new_markdown_cell(f"### `{image_name}`\n\n![](plots/{image_name})"))

    cells.extend(
        [
            nbformat.v4.new_markdown_cell(
                "## 9. 직접 필터링하기\n\n"
                "아래 셀들은 사용자가 직접 feature set, comparison, feature 이름으로 필터링할 때 쓰는 셀입니다."
            ),
            nbformat.v4.new_code_cell(
                "feature_set = 'all_current'\n"
                "metrics[metrics['feature_set'] == feature_set]\n"
            ),
            nbformat.v4.new_code_cell(
                "feature = 'has_additional_A'\n"
                "statistics[statistics['feature'].eq(feature)].sort_values(\n"
                "    ['comparison', 'mann_whitney_u_pvalue', 'separation_score'],\n"
                "    ascending=[True, True, False]\n"
                ")\n"
            ),
            nbformat.v4.new_code_cell(
                "feature_set = 'all_current'\n"
                "importances[importances['feature_set'].eq(feature_set)].head(30)\n"
            ),
        ]
    )

    notebook = nbformat.v4.new_notebook()
    notebook.cells = cells
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    notebook_path = output_dir / "feature_experiment_report.ipynb"
    with notebook_path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    return notebook_path


def run_feature_experiments(
    *,
    run_version: str,
    project_root: str | Path = ".",
    config_path: str | Path = "config/randomforest.yaml",
    output_root: str | Path = "outputs/feature_experiments",
    overwrite: bool = False,
    max_plotted_features: int = 12,
) -> dict:
    root = Path(project_root).resolve()
    normalized = normalize_run_version(run_version)
    output_dir = root / output_root / f"run_{normalized}"
    plots_dir = output_dir / "plots"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Feature experiment output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    paths = _run_paths(root, normalized)
    artifact = load_official_model(normalized, root)
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = load_config(config_file)

    train_frame, test_frame, csv_paths = _rebuild_evaluation_frames_from_metrics(root, metrics)
    base_features = list(artifact["feature_columns"])
    feature_sets = build_feature_sets(base_features)
    random_seed = int(metrics["random_seed"])
    model_params = _base_model_params(artifact, config, random_seed)

    y_train = train_frame[LABEL_COLUMN].astype("int8")
    y_test = test_frame[LABEL_COLUMN].astype("int8")
    metrics_rows: list[dict] = []
    confusion_rows: list[dict] = []
    importance_rows: list[dict] = []
    probability_rows: list[dict] = []
    statistics_rows: list[dict] = []
    plotted_once = False

    for feature_set, features in feature_sets.items():
        missing = [feature for feature in features if feature not in train_frame.columns]
        if missing:
            raise ValueError(f"{feature_set} has features missing from processed CSV: {missing}")
        x_train = train_frame[features].apply(pd.to_numeric, errors="coerce").astype("float32")
        x_test = test_frame[features].apply(pd.to_numeric, errors="coerce").astype("float32")
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(**model_params)),
            ]
        )
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        y_pred = (probabilities >= FIXED_THRESHOLD).astype("int8")

        metric = _metric_row(
            feature_set=feature_set,
            feature_count=len(features),
            y_true=y_test,
            y_pred=y_pred,
            probabilities=probabilities,
        )
        metrics_rows.append(metric)
        confusion_rows.append(
            {
                "feature_set": feature_set,
                "TN": metric["TN"],
                "FP": metric["FP"],
                "FN": metric["FN"],
                "TP": metric["TP"],
            }
        )
        importances = pd.DataFrame(
            {
                "feature_set": feature_set,
                "feature": features,
                "importance": model.named_steps["model"].feature_importances_,
            }
        ).sort_values(["feature_set", "importance"], ascending=[True, False])
        importance_rows.extend(importances.to_dict("records"))

        analysis_features = [
            feature
            for feature in list(dict.fromkeys(TOP_IMPORTANCE_FEATURES + DROPPED_SHORTCUT_FEATURES))
            if feature in features
        ]
        if not analysis_features:
            analysis_features = importances.head(max_plotted_features)["feature"].tolist()
        statistics_rows.extend(
            _statistic_rows(
                feature_set=feature_set,
                test_frame=test_frame,
                features=analysis_features,
                y_true=y_test,
                y_pred=y_pred,
            )
        )
        probability_rows.extend(_probability_rows(feature_set, y_test, y_pred, probabilities))

        _plot_confusion(metric, plots_dir)
        _plot_importance(importances, feature_set, plots_dir)
        _plot_probability_distribution(
            feature_set,
            y_test,
            y_pred,
            probabilities,
            plots_dir,
            filename=f"probability_distribution_{_safe_name(feature_set)}.png",
        )
        if feature_set == "all_current" or not plotted_once:
            _plot_feature_distributions(
                test_frame,
                feature_set=feature_set,
                features=analysis_features[:max_plotted_features],
                y_true=y_test,
                y_pred=y_pred,
                plots_dir=plots_dir,
            )
            _plot_probability_distribution(
                feature_set,
                y_test,
                y_pred,
                probabilities,
                plots_dir,
                filename="probability_distribution.png",
            )
            plotted_once = True

    metrics_frame = pd.DataFrame(metrics_rows).sort_values("feature_set")
    confusion_frame = pd.DataFrame(confusion_rows).sort_values("feature_set")
    importance_frame = pd.DataFrame(importance_rows).sort_values(
        ["feature_set", "importance"], ascending=[True, False]
    )
    statistics_frame = pd.DataFrame(statistics_rows)
    probability_frame = pd.DataFrame(probability_rows)

    metrics_path = output_dir / "metrics_summary.csv"
    confusion_path = output_dir / "confusion_matrices.csv"
    importance_path = output_dir / "feature_importances.csv"
    statistics_path = output_dir / "feature_statistics.csv"
    probability_path = output_dir / "probability_summary.csv"
    metrics_frame.to_csv(metrics_path, index=False)
    confusion_frame.to_csv(confusion_path, index=False)
    importance_frame.to_csv(importance_path, index=False)
    statistics_frame.to_csv(statistics_path, index=False)
    probability_frame.to_csv(probability_path, index=False)
    _plot_metrics(metrics_frame, plots_dir)

    experiment_config = {
        "analysis_type": "feature_experiments",
        "base_run_version": normalized,
        "base_model_path": str(paths["model"]),
        "threshold": FIXED_THRESHOLD,
        "class_weight": artifact.get("best_params", {}).get("model__class_weight"),
        "model_type": "RandomForestClassifier",
        "model_params": model_params,
        "dataset_versions": metrics.get("dataset_versions"),
        "train_dataset_versions": metrics.get("train_dataset_versions"),
        "test_dataset_versions": metrics.get("test_dataset_versions"),
        "evaluation_strategy": metrics.get("evaluation_strategy"),
        "class_ratio": metrics.get("class_ratio"),
        "source_processed_csv_files": [str(path) for path in csv_paths],
        "train_row_count": int(len(train_frame)),
        "test_row_count": int(len(test_frame)),
        "class_distribution_train": _class_counts(train_frame),
        "class_distribution_test": _class_counts(test_frame),
        "base_feature_columns": base_features,
        "feature_sets": feature_sets,
        "dropped_shortcut_features": DROPPED_SHORTCUT_FEATURES,
        "top_importance_features": TOP_IMPORTANCE_FEATURES,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "metrics_summary": str(metrics_path),
            "confusion_matrices": str(confusion_path),
            "feature_importances": str(importance_path),
            "feature_statistics": str(statistics_path),
            "probability_summary": str(probability_path),
            "plots_dir": str(plots_dir),
        },
    }
    config_output = output_dir / "config.yaml"
    _write_yaml(config_output, experiment_config)
    notebook_path = write_feature_experiment_notebook(output_dir)
    return {
        "output_dir": output_dir,
        "config_path": config_output,
        "notebook_path": notebook_path,
        "metrics_path": metrics_path,
        "confusion_path": confusion_path,
        "importance_path": importance_path,
        "statistics_path": statistics_path,
        "probability_path": probability_path,
        "plots_dir": plots_dir,
    }
