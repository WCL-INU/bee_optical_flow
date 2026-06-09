from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_FEATURES = Path("analysis") / "video_features" / "output" / "video_image_flow_features.csv"
DEFAULT_TRUTH = Path("validation") / "data" / "merged_data.xlsx"
DEFAULT_MODELS = Path("validation") / "output" / "regression_model_comparison.csv"
DEFAULT_OUTPUT_DIR = Path("analysis") / "video_features" / "output"
DEFAULT_MERGED_NAME = "video_features_with_linear_errors.csv"
DEFAULT_CORRELATION_NAME = "feature_error_correlations.csv"
DEFAULT_COVARIANCE_NAME = "feature_error_covariances.csv"
DEFAULT_FEATURE_CORRELATION_NAME = "feature_feature_correlations.csv"
DEFAULT_FEATURE_GROUP_NAME = "feature_correlation_groups.csv"
DEFAULT_REPRESENTATIVE_NAME = "representative_features.csv"
DEFAULT_REPRESENTATIVE_SCATTER_NAME = "representative_feature_error_scatter.png"
DEFAULT_COMBINED_MODEL_NAME = "representative_feature_error_models.csv"
DEFAULT_COMBINED_PREDICTION_NAME = "representative_feature_error_predictions.csv"
DEFAULT_COMBINED_SCATTER_NAME = "representative_feature_error_model_scatter.png"
NORMALIZATION_CHOICES = ("none", "zscore", "minmax", "robust")
OUTLIER_FILTER_CHOICES = ("none", "iqr", "zscore")

DEFAULT_META_EXCLUDES = {
    "fps",
    "width",
    "height",
    "frame_count",
    "video_duration_sec",
    "processed_frame_pairs",
    "frame_stride",
    "roi_x1",
    "roi_y1",
    "roi_x2",
    "roi_y2",
    "ent_x1",
    "ent_y1",
    "ent_x2",
    "ent_y2",
    "entrance_roi_x1",
    "entrance_roi_y1",
    "entrance_roi_x2",
    "entrance_roi_y2",
    "entrance_area_px",
    "roi_area_px",
    "boundary_band_area_px",
    "boundary_band_px",
    "blur_kernel",
    "flow_mag_threshold",
    "normal_flow_threshold",
    "persist_decay",
    "persist_threshold",
    "min_flow_component_area",
    "balance_ratio_threshold",
    "processing_time_sec",
}


@dataclass(frozen=True)
class ModelSpec:
    target_col: str
    x_col: str
    model: str
    slope: float
    intercept: float


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table type: {path}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".xlsx":
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)


def normalize_datetime_key(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
    result = df.copy()
    if datetime_col not in result.columns:
        raise KeyError(f"Missing datetime column: {datetime_col}")
    result["datetime_key"] = pd.to_datetime(result[datetime_col], errors="coerce")
    return result


def parse_video_datetime(video_name: str) -> pd.Timestamp | pd.NaT:
    match = re.search(r"ANU-25-summer-\d+_(\d{8})_(\d{6})", str(video_name))
    if not match:
        return pd.NaT
    return pd.to_datetime(match.group(1) + match.group(2), format="%Y%m%d%H%M%S", errors="coerce")


def ensure_feature_datetime(features: pd.DataFrame) -> pd.DataFrame:
    if "datetime" in features.columns:
        return normalize_datetime_key(features, "datetime")
    if "video" not in features.columns:
        raise KeyError("Feature table needs either 'datetime' or 'video' to build a datetime key.")
    result = features.copy()
    result["datetime_key"] = result["video"].map(parse_video_datetime)
    return result


def load_linear_model_specs(models: pd.DataFrame, model_name: str = "linear") -> dict[str, ModelSpec]:
    required = {"model", "status", "x_col", "y_col", "slope", "intercept"}
    missing = required - set(models.columns)
    if missing:
        raise KeyError(f"Model table is missing columns: {sorted(missing)}")

    rows = models[(models["model"] == model_name) & (models["status"] == "ok")]
    specs: dict[str, ModelSpec] = {}
    for _, row in rows.iterrows():
        target = str(row["y_col"])
        specs[target] = ModelSpec(
            target_col=target,
            x_col=str(row["x_col"]),
            model=str(row["model"]),
            slope=float(row["slope"]),
            intercept=float(row["intercept"]),
        )
    return specs


def add_model_errors(
    truth: pd.DataFrame,
    model_specs: dict[str, ModelSpec],
    targets: Iterable[str] | None = None,
) -> pd.DataFrame:
    result = truth.copy()
    selected_targets = list(targets) if targets is not None else list(model_specs)

    for target in selected_targets:
        if target not in model_specs:
            raise KeyError(f"No model spec found for target: {target}")
        spec = model_specs[target]
        if spec.x_col not in result.columns:
            raise KeyError(f"Truth table is missing model input column: {spec.x_col}")
        if spec.target_col not in result.columns:
            raise KeyError(f"Truth table is missing target column: {spec.target_col}")

        pred_col = f"{target}_pred"
        signed_col = f"{target}_signed_error"
        abs_col = f"{target}_abs_error"
        result[pred_col] = spec.intercept + spec.slope * pd.to_numeric(result[spec.x_col], errors="coerce")
        result[signed_col] = result[pred_col] - pd.to_numeric(result[spec.target_col], errors="coerce")
        result[abs_col] = result[signed_col].abs()

    if {"in", "out"}.issubset(selected_targets):
        result["total_actual"] = pd.to_numeric(result["in"], errors="coerce") + pd.to_numeric(
            result["out"], errors="coerce"
        )
        result["total_pred"] = result["in_pred"] + result["out_pred"]
        result["total_signed_error"] = result["total_pred"] - result["total_actual"]
        result["total_abs_error"] = result["total_signed_error"].abs()
        result["sum_abs_error"] = result["in_abs_error"] + result["out_abs_error"]

    return result


def build_feature_error_table(
    features: pd.DataFrame,
    truth: pd.DataFrame,
    models: pd.DataFrame,
    *,
    devices: Iterable[int] | None = None,
    model_name: str = "linear",
    target_cols: Iterable[str] | None = None,
    join_cols: tuple[str, ...] = ("datetime_key", "device"),
) -> pd.DataFrame:
    feature_keyed = ensure_feature_datetime(features)
    truth_keyed = normalize_datetime_key(truth, "datetime")

    if devices is not None:
        device_set = {int(device) for device in devices}
        feature_keyed = feature_keyed[feature_keyed["device"].astype("Int64").isin(device_set)]
        truth_keyed = truth_keyed[truth_keyed["device"].astype("Int64").isin(device_set)]

    model_specs = load_linear_model_specs(models, model_name=model_name)
    truth_with_errors = add_model_errors(truth_keyed, model_specs, targets=target_cols)
    truth_keep = [
        col
        for col in truth_with_errors.columns
        if col in set(join_cols)
        or col.endswith("_pred")
        or col.endswith("_signed_error")
        or col.endswith("_abs_error")
        or col in {"in", "out", "total_actual", "total_pred", "total_signed_error", "total_abs_error", "sum_abs_error"}
    ]
    return feature_keyed.merge(
        truth_with_errors[truth_keep],
        on=list(join_cols),
        how="inner",
        suffixes=("", "_truth"),
    )


def merge_extra_tables(
    base: pd.DataFrame,
    extra_tables: Iterable[pd.DataFrame],
    *,
    join_cols: Iterable[str] = ("datetime_key", "device"),
    how: str = "left",
) -> pd.DataFrame:
    result = base.copy()
    for index, table in enumerate(extra_tables, start=1):
        extra = table.copy()
        if "datetime_key" in join_cols and "datetime_key" not in extra.columns:
            extra = normalize_datetime_key(extra, "datetime")
        result = result.merge(extra, on=list(join_cols), how=how, suffixes=("", f"_extra{index}"))
    return result


def numeric_columns(
    df: pd.DataFrame,
    *,
    include_regex: str | None = None,
    exclude_cols: Iterable[str] = (),
    min_non_null: int = 10,
    require_variance: bool = True,
) -> list[str]:
    include_pattern = re.compile(include_regex) if include_regex else None
    excluded = set(exclude_cols)
    columns: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        if include_pattern and not include_pattern.search(col):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        values = df[col].replace([np.inf, -np.inf], np.nan)
        if values.notna().sum() < min_non_null:
            continue
        if require_variance and values.nunique(dropna=True) <= 1:
            continue
        columns.append(col)
    return columns


def filter_xy_outliers(
    xy: pd.DataFrame,
    *,
    method: str = "none",
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    if method == "none" or xy.empty:
        return xy
    if method not in OUTLIER_FILTER_CHOICES:
        raise ValueError(f"Unsupported outlier filter: {method}")

    keep = pd.Series(True, index=xy.index)
    for col in xy.columns:
        values = xy[col]
        if values.nunique(dropna=True) <= 1:
            continue
        if method == "iqr":
            q25 = values.quantile(0.25)
            q75 = values.quantile(0.75)
            iqr = q75 - q25
            if not math.isfinite(iqr) or abs(iqr) < 1e-12:
                continue
            lower = q25 - iqr_multiplier * iqr
            upper = q75 + iqr_multiplier * iqr
            keep &= values.between(lower, upper)
        elif method == "zscore":
            std = values.std(ddof=1)
            if not math.isfinite(std) or abs(std) < 1e-12:
                continue
            keep &= ((values - values.mean()).abs() / std) <= z_threshold
    return xy[keep]


def pairwise_correlations(
    df: pd.DataFrame,
    x_cols: Iterable[str],
    y_cols: Iterable[str],
    *,
    methods: Iterable[str] = ("pearson", "spearman"),
    min_n: int = 10,
    outlier_filter: str = "none",
    outlier_iqr_multiplier: float = 1.5,
    outlier_z_threshold: float = 3.0,
) -> pd.DataFrame:
    if outlier_filter not in OUTLIER_FILTER_CHOICES:
        raise ValueError(f"Unsupported outlier filter: {outlier_filter}")

    rows: list[dict[str, object]] = []
    for y_col in y_cols:
        if y_col not in df.columns:
            raise KeyError(f"Missing target column: {y_col}")
        y = pd.to_numeric(df[y_col], errors="coerce")
        for x_col in x_cols:
            if x_col not in df.columns:
                raise KeyError(f"Missing feature column: {x_col}")
            x = pd.to_numeric(df[x_col], errors="coerce")
            xy = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
            n_raw = len(xy)
            xy = filter_xy_outliers(
                xy,
                method=outlier_filter,
                iqr_multiplier=outlier_iqr_multiplier,
                z_threshold=outlier_z_threshold,
            )
            row: dict[str, object] = {
                "target": y_col,
                "feature": x_col,
                "n": len(xy),
                "n_raw": n_raw,
                "outlier_filter": outlier_filter,
            }
            if len(xy) < min_n or xy.iloc[:, 0].nunique() <= 1 or xy.iloc[:, 1].nunique() <= 1:
                for method in methods:
                    row[f"{method}_r"] = math.nan
                    row[f"{method}_abs_r"] = math.nan
                rows.append(row)
                continue

            for method in methods:
                r_value = float(xy.corr(method=method).iloc[0, 1])
                row[f"{method}_r"] = r_value
                row[f"{method}_abs_r"] = abs(r_value)
            rows.append(row)
    return pd.DataFrame(rows)


def normalize_series(values: pd.Series, method: str) -> pd.Series:
    if method == "none":
        return values
    if method == "zscore":
        std = values.std(ddof=1)
        if not math.isfinite(std) or abs(std) < 1e-12:
            return pd.Series(np.nan, index=values.index)
        return (values - values.mean()) / std
    if method == "minmax":
        min_value = values.min()
        max_value = values.max()
        span = max_value - min_value
        if not math.isfinite(span) or abs(span) < 1e-12:
            return pd.Series(np.nan, index=values.index)
        return (values - min_value) / span
    if method == "robust":
        q25 = values.quantile(0.25)
        q75 = values.quantile(0.75)
        iqr = q75 - q25
        if not math.isfinite(iqr) or abs(iqr) < 1e-12:
            return pd.Series(np.nan, index=values.index)
        return (values - values.median()) / iqr
    raise ValueError(f"Unsupported normalization: {method}")


def pairwise_covariances(
    df: pd.DataFrame,
    x_cols: Iterable[str],
    y_cols: Iterable[str],
    *,
    min_n: int = 10,
    ddof: int = 1,
    normalization: str = "none",
) -> pd.DataFrame:
    if normalization not in NORMALIZATION_CHOICES:
        raise ValueError(f"Unsupported normalization: {normalization}")

    rows: list[dict[str, object]] = []
    for y_col in y_cols:
        if y_col not in df.columns:
            raise KeyError(f"Missing target column: {y_col}")
        y = pd.to_numeric(df[y_col], errors="coerce")
        for x_col in x_cols:
            if x_col not in df.columns:
                raise KeyError(f"Missing feature column: {x_col}")
            x = pd.to_numeric(df[x_col], errors="coerce")
            xy = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
            row: dict[str, object] = {
                "target": y_col,
                "feature": x_col,
                "n": len(xy),
                "normalization": normalization,
            }
            if len(xy) <= ddof or len(xy) < min_n or xy.iloc[:, 0].nunique() <= 1 or xy.iloc[:, 1].nunique() <= 1:
                row.update(
                    {
                        "covariance": math.nan,
                        "abs_covariance": math.nan,
                        "feature_mean": math.nan,
                        "target_mean": math.nan,
                        "feature_std": math.nan,
                        "target_std": math.nan,
                    }
                )
                rows.append(row)
                continue

            x_values = xy.iloc[:, 0]
            y_values = xy.iloc[:, 1]
            x_used = normalize_series(x_values, normalization)
            y_used = normalize_series(y_values, normalization)
            used = pd.concat([x_used, y_used], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
            if len(used) <= ddof or used.iloc[:, 0].nunique() <= 1 or used.iloc[:, 1].nunique() <= 1:
                covariance = math.nan
                feature_mean = math.nan
                target_mean = math.nan
                feature_std = math.nan
                target_std = math.nan
            else:
                covariance = float(used.iloc[:, 0].cov(used.iloc[:, 1], ddof=ddof))
                feature_mean = float(used.iloc[:, 0].mean())
                target_mean = float(used.iloc[:, 1].mean())
                feature_std = float(used.iloc[:, 0].std(ddof=ddof))
                target_std = float(used.iloc[:, 1].std(ddof=ddof))
            row.update(
                {
                    "covariance": covariance,
                    "abs_covariance": abs(covariance),
                    "feature_mean": feature_mean,
                    "target_mean": target_mean,
                    "feature_std": feature_std,
                    "target_std": target_std,
                    "feature_original_mean": float(x_values.mean()),
                    "target_original_mean": float(y_values.mean()),
                    "feature_original_std": float(x_values.std(ddof=ddof)),
                    "target_original_std": float(y_values.std(ddof=ddof)),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def top_correlations(
    correlations: pd.DataFrame,
    *,
    target: str | None = None,
    method: str = "pearson",
    n: int = 20,
) -> pd.DataFrame:
    result = correlations
    if target is not None:
        result = result[result["target"] == target]
    return result.sort_values(f"{method}_abs_r", ascending=False).head(n)


def top_covariances(
    covariances: pd.DataFrame,
    *,
    target: str | None = None,
    n: int = 20,
) -> pd.DataFrame:
    result = covariances
    if target is not None:
        result = result[result["target"] == target]
    return result.sort_values("abs_covariance", ascending=False).head(n)


def feature_feature_correlations(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    *,
    methods: Iterable[str] = ("pearson", "spearman"),
    min_n: int = 10,
) -> pd.DataFrame:
    cols = list(feature_cols)
    if len(cols) < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "n"])

    data = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = data.notna().astype(np.int16)
    counts = valid.T.dot(valid)
    upper = np.triu_indices(len(cols), k=1)

    result = pd.DataFrame(
        {
            "feature_a": [cols[index] for index in upper[0]],
            "feature_b": [cols[index] for index in upper[1]],
            "n": counts.to_numpy()[upper],
        }
    )
    for method in methods:
        corr = data.corr(method=method).to_numpy()
        values = corr[upper]
        values = np.where(result["n"].to_numpy() >= min_n, values, np.nan)
        result[f"{method}_r"] = values
        result[f"{method}_abs_r"] = np.abs(values)

    sort_col = f"{next(iter(methods))}_abs_r"
    return result.sort_values(sort_col, ascending=False)


def top_feature_correlations(
    feature_correlations: pd.DataFrame,
    *,
    method: str = "pearson",
    n: int = 20,
    max_abs_r: float | None = None,
) -> pd.DataFrame:
    result = feature_correlations
    if max_abs_r is not None:
        result = result[result[f"{method}_abs_r"] <= max_abs_r]
    return result.sort_values(f"{method}_abs_r", ascending=False).head(n)


def group_correlated_features(
    feature_cols: Iterable[str],
    feature_correlations: pd.DataFrame,
    *,
    method: str = "pearson",
    threshold: float = 0.95,
) -> dict[str, int]:
    cols = list(feature_cols)
    parent = {col: col for col in cols}

    def find(col: str) -> str:
        while parent[col] != col:
            parent[col] = parent[parent[col]]
            col = parent[col]
        return col

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    value_col = f"{method}_abs_r"
    similar = feature_correlations[feature_correlations[value_col] >= threshold]
    for _, row in similar.iterrows():
        union(str(row["feature_a"]), str(row["feature_b"]))

    roots = {col: find(col) for col in cols}
    root_to_id = {root: index for index, root in enumerate(sorted(set(roots.values())), start=1)}
    return {col: root_to_id[root] for col, root in roots.items()}


def choose_representative_features(
    feature_cols: Iterable[str],
    feature_to_group: dict[str, int],
    target_correlations: pd.DataFrame,
    *,
    method: str = "pearson",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_col = f"{method}_abs_r"
    signed_col = f"{method}_r"
    score_rows = []
    for feature in feature_cols:
        rows = target_correlations[target_correlations["feature"] == feature].dropna(subset=[score_col])
        if rows.empty:
            score_rows.append(
                {
                    "feature": feature,
                    "group_id": feature_to_group[feature],
                    "best_target": "",
                    "best_abs_r": math.nan,
                    "best_r": math.nan,
                    "mean_abs_r": math.nan,
                    "non_null_targets": 0,
                }
            )
            continue
        best = rows.sort_values(score_col, ascending=False).iloc[0]
        score_rows.append(
            {
                "feature": feature,
                "group_id": feature_to_group[feature],
                "best_target": best["target"],
                "best_abs_r": float(best[score_col]),
                "best_r": float(best[signed_col]),
                "mean_abs_r": float(rows[score_col].mean()),
                "non_null_targets": int(rows[score_col].notna().sum()),
            }
        )

    scores = pd.DataFrame(score_rows)
    group_sizes = scores.groupby("group_id").size().rename("group_size")
    members = scores.groupby("group_id")["feature"].apply(lambda values: "|".join(sorted(values))).rename("members")
    representatives = (
        scores.sort_values(["group_id", "best_abs_r", "mean_abs_r", "feature"], ascending=[True, False, False, True])
        .groupby("group_id", as_index=False)
        .first()
        .merge(group_sizes, on="group_id", how="left")
    )
    groups = representatives.merge(members, on="group_id", how="left")
    groups = groups[
        [
            "group_id",
            "group_size",
            "feature",
            "best_target",
            "best_abs_r",
            "best_r",
            "mean_abs_r",
            "members",
        ]
    ].rename(columns={"feature": "representative_feature"})
    representatives = representatives.rename(columns={"feature": "representative_feature"})
    return groups.sort_values(["group_size", "best_abs_r"], ascending=[False, False]), representatives


def representative_scatter_pairs(
    target_correlations: pd.DataFrame,
    representatives: pd.DataFrame,
    *,
    method: str = "pearson",
    n: int = 20,
) -> list[tuple[str, str]]:
    rep_features = set(representatives["representative_feature"])
    rows = target_correlations[target_correlations["feature"].isin(rep_features)].dropna(subset=[f"{method}_abs_r"])
    rows = rows.sort_values(f"{method}_abs_r", ascending=False).head(n)
    return [(str(row["feature"]), str(row["target"])) for _, row in rows.iterrows()]


def make_scatter_grid(
    df: pd.DataFrame,
    pairs: Iterable[tuple[str, str]],
    output_path: Path,
    *,
    method: str = "pearson",
    outlier_filter: str = "none",
    outlier_iqr_multiplier: float = 1.5,
    outlier_z_threshold: float = 3.0,
) -> None:
    pairs = list(pairs)
    if not pairs:
        return
    cols = min(3, len(pairs))
    rows = math.ceil(len(pairs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for ax in axes.reshape(-1):
        ax.axis("off")

    for ax, (x_col, y_col) in zip(axes.reshape(-1), pairs):
        data = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
        data = filter_xy_outliers(
            data,
            method=outlier_filter,
            iqr_multiplier=outlier_iqr_multiplier,
            z_threshold=outlier_z_threshold,
        )
        ax.axis("on")
        ax.scatter(data[x_col], data[y_col], s=14, alpha=0.65)
        r_value = data.corr(method=method).iloc[0, 1] if len(data) >= 2 else math.nan
        ax.set_title(f"{method} r = {r_value:.3f}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def clean_xy_for_model(df: pd.DataFrame, feature_cols: list[str], target: str) -> tuple[pd.DataFrame, pd.Series]:
    data = df[feature_cols + [target]].replace([np.inf, -np.inf], np.nan)
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna(subset=[target])
    return data[feature_cols].apply(pd.to_numeric, errors="coerce"), data[target]


def standardize_with_train(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    means = x_train.mean()
    stds = x_train.std(ddof=1).replace(0, np.nan)
    x_train_filled = x_train.fillna(means)
    x_test_filled = x_test.fillna(means)
    x_train_scaled = ((x_train_filled - means) / stds).fillna(0.0).to_numpy(dtype=float)
    x_test_scaled = ((x_test_filled - means) / stds).fillna(0.0).to_numpy(dtype=float)
    return x_train_scaled, x_test_scaled, means, stds


def fit_ridge_with_intercept(x_values: np.ndarray, y_values: np.ndarray, alpha: float) -> np.ndarray:
    design = np.column_stack([np.ones(len(x_values)), x_values])
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    return np.linalg.pinv(design.T @ design + penalty) @ design.T @ y_values


def predict_with_intercept(x_values: np.ndarray, beta: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x_values)), x_values])
    return design @ beta


def safe_corr(left: np.ndarray | pd.Series, right: np.ndarray | pd.Series) -> float:
    data = pd.DataFrame({"left": left, "right": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 2 or data["left"].nunique() <= 1 or data["right"].nunique() <= 1:
        return math.nan
    return float(data["left"].corr(data["right"]))


def cross_validated_ridge_predictions(
    x_values: pd.DataFrame,
    y_values: pd.Series,
    *,
    alpha: float = 1.0,
    folds: int = 5,
    seed: int = 7,
) -> np.ndarray:
    n = len(y_values)
    if n < 2:
        return np.full(n, math.nan)
    folds = max(2, min(int(folds), n))
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    fold_ids = np.array_split(indices, folds)
    predictions = np.full(n, math.nan)

    for test_idx in fold_ids:
        train_idx = np.setdiff1d(indices, test_idx, assume_unique=False)
        x_train = x_values.iloc[train_idx]
        x_test = x_values.iloc[test_idx]
        y_train = y_values.iloc[train_idx].to_numpy(dtype=float)
        x_train_scaled, x_test_scaled, _, _ = standardize_with_train(x_train, x_test)
        beta = fit_ridge_with_intercept(x_train_scaled, y_train, alpha)
        predictions[test_idx] = predict_with_intercept(x_test_scaled, beta)
    return predictions


def evaluate_representative_combinations(
    df: pd.DataFrame,
    representatives: pd.DataFrame,
    target_correlations: pd.DataFrame,
    targets: Iterable[str],
    *,
    method: str = "pearson",
    top_features: int = 20,
    alpha: float = 10.0,
    folds: int = 5,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_col = f"{method}_abs_r"
    signed_col = f"{method}_r"
    rep_features = set(representatives["representative_feature"])
    model_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for target in targets:
        target_rows = target_correlations[
            target_correlations["feature"].isin(rep_features) & (target_correlations["target"] == target)
        ].dropna(subset=[score_col]).sort_values(score_col, ascending=False)
        if target_rows.empty:
            continue
        selected = [str(feature) for feature in target_rows["feature"].head(top_features)]
        x_values, y_values = clean_xy_for_model(df, selected, target)
        if len(y_values) < max(10, folds):
            continue

        cv_pred = cross_validated_ridge_predictions(
            x_values,
            y_values,
            alpha=alpha,
            folds=folds,
            seed=seed,
        )
        x_scaled, _, _, _ = standardize_with_train(x_values, x_values)
        beta = fit_ridge_with_intercept(x_scaled, y_values.to_numpy(dtype=float), alpha)
        fitted_pred = predict_with_intercept(x_scaled, beta)

        best_single = target_rows.iloc[0]
        cv_r = safe_corr(cv_pred, y_values)
        fitted_r = safe_corr(fitted_pred, y_values)
        residual = y_values.to_numpy(dtype=float) - cv_pred
        rmse = float(np.sqrt(np.nanmean(residual * residual)))
        mae = float(np.nanmean(np.abs(residual)))
        model_rows.append(
            {
                "target": target,
                "n": len(y_values),
                "selected_feature_count": len(selected),
                "alpha": alpha,
                "folds": folds,
                "best_single_feature": best_single["feature"],
                "best_single_r": float(best_single[signed_col]),
                "best_single_abs_r": float(best_single[score_col]),
                "cv_r": cv_r,
                "cv_abs_r": abs(cv_r) if pd.notna(cv_r) else math.nan,
                "fitted_r": fitted_r,
                "fitted_abs_r": abs(fitted_r) if pd.notna(fitted_r) else math.nan,
                "cv_rmse": rmse,
                "cv_mae": mae,
                "selected_features": "|".join(selected),
            }
        )

        pred_frame = pd.DataFrame(
            {
                "target": target,
                "actual": y_values.to_numpy(dtype=float),
                "cv_prediction": cv_pred,
                "fitted_prediction": fitted_pred,
            },
            index=y_values.index,
        )
        prediction_frames.append(pred_frame.reset_index(names="source_index"))

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return pd.DataFrame(model_rows).sort_values("cv_abs_r", ascending=False), predictions


def make_prediction_scatter(predictions: pd.DataFrame, output_path: Path) -> None:
    if predictions.empty:
        return
    targets = list(predictions["target"].drop_duplicates())
    cols = min(3, len(targets))
    rows = math.ceil(len(targets) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for ax in axes.reshape(-1):
        ax.axis("off")
    for ax, target in zip(axes.reshape(-1), targets):
        data = predictions[predictions["target"] == target].dropna(subset=["actual", "cv_prediction"])
        ax.axis("on")
        ax.scatter(data["cv_prediction"], data["actual"], s=14, alpha=0.65)
        r_value = safe_corr(data["cv_prediction"], data["actual"])
        ax.set_title(f"{target}: CV r = {r_value:.3f}")
        ax.set_xlabel("combined feature prediction")
        ax.set_ylabel(target)
        ax.grid(True, alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def default_error_targets(df: pd.DataFrame) -> list[str]:
    preferred = ["total_abs_error", "sum_abs_error", "in_abs_error", "out_abs_error"]
    return [col for col in preferred if col in df.columns]


def parse_int_list(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
    devices: list[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                devices.append(int(part))
    return devices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join video features, validation counts, regression predictions, and pairwise correlations."
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--merged-name", default=DEFAULT_MERGED_NAME)
    parser.add_argument("--correlation-name", default=DEFAULT_CORRELATION_NAME)
    parser.add_argument("--covariance-name", default=DEFAULT_COVARIANCE_NAME)
    parser.add_argument("--feature-correlation-name", default=DEFAULT_FEATURE_CORRELATION_NAME)
    parser.add_argument("--feature-group-name", default=DEFAULT_FEATURE_GROUP_NAME)
    parser.add_argument("--representative-name", default=DEFAULT_REPRESENTATIVE_NAME)
    parser.add_argument("--representative-scatter-name", default=DEFAULT_REPRESENTATIVE_SCATTER_NAME)
    parser.add_argument("--combined-model-name", default=DEFAULT_COMBINED_MODEL_NAME)
    parser.add_argument("--combined-prediction-name", default=DEFAULT_COMBINED_PREDICTION_NAME)
    parser.add_argument("--combined-scatter-name", default=DEFAULT_COMBINED_SCATTER_NAME)
    parser.add_argument("--device", nargs="+", help="One or more device ids, e.g. --device 3 or --device 3,5")
    parser.add_argument("--model", default="linear")
    parser.add_argument("--targets", nargs="+", help="Error target columns to analyze. Defaults to common abs-error columns.")
    parser.add_argument("--feature-regex", help="Only analyze feature columns matching this regex.")
    parser.add_argument("--include-meta", action="store_true", help="Include mostly static metadata columns in correlation scans.")
    parser.add_argument("--min-n", type=int, default=10)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--method", choices=["pearson", "spearman"], default="pearson")
    parser.add_argument(
        "--outlier-filter",
        choices=OUTLIER_FILTER_CHOICES,
        default="none",
        help="Remove pairwise outliers before target-feature correlation calculation.",
    )
    parser.add_argument("--outlier-iqr-multiplier", type=float, default=1.5)
    parser.add_argument("--outlier-z-threshold", type=float, default=3.0)
    parser.add_argument("--feature-correlations", action="store_true", help="Also write feature-to-feature correlations.")
    parser.add_argument(
        "--group-features",
        action="store_true",
        help="Group highly correlated features and choose one representative feature per group.",
    )
    parser.add_argument(
        "--feature-group-threshold",
        type=float,
        default=0.95,
        help="Absolute feature-feature correlation threshold for grouping.",
    )
    parser.add_argument(
        "--representative-scatter",
        action="store_true",
        help="Write scatter plots for representative feature/error pairs.",
    )
    parser.add_argument(
        "--combine-representatives",
        action="store_true",
        help="Fit cross-validated ridge models that combine representative features for each target.",
    )
    parser.add_argument("--combo-top-features", type=int, default=20)
    parser.add_argument("--combo-alpha", type=float, default=10.0)
    parser.add_argument("--combo-folds", type=int, default=5)
    parser.add_argument(
        "--feature-correlation-max-abs",
        type=float,
        default=0.999,
        help="When printing feature-feature correlations, also show pairs below this absolute r.",
    )
    parser.add_argument("--covariance", action="store_true", help="Also write and print pairwise covariances.")
    parser.add_argument(
        "--covariance-normalization",
        choices=NORMALIZATION_CHOICES,
        default="none",
        help="Normalize each feature and target before covariance calculation.",
    )
    parser.add_argument("--scatter", action="store_true", help="Write a scatter grid for the strongest pairs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    devices = parse_int_list(args.device)
    features = read_table(args.features)
    truth = read_table(args.truth)
    models = read_table(args.models)

    merged = build_feature_error_table(
        features,
        truth,
        models,
        devices=devices,
        model_name=args.model,
    )
    targets = args.targets or default_error_targets(merged)
    if not targets:
        raise ValueError("No target columns were found. Pass --targets explicitly.")

    excluded = set(targets) | {
        "device",
        "time",
        "hour",
        "in",
        "out",
        "total_actual",
        "total_pred",
        "total_signed_error",
        "total_abs_error",
        "sum_abs_error",
    }
    excluded.update(col for col in merged.columns if col.endswith("_pred") or col.endswith("_error"))
    if not args.include_meta:
        excluded.update(DEFAULT_META_EXCLUDES)

    features_to_scan = numeric_columns(
        merged,
        include_regex=args.feature_regex,
        exclude_cols=excluded,
        min_non_null=args.min_n,
    )
    correlations = pairwise_correlations(
        merged,
        features_to_scan,
        targets,
        min_n=args.min_n,
        outlier_filter=args.outlier_filter,
        outlier_iqr_multiplier=args.outlier_iqr_multiplier,
        outlier_z_threshold=args.outlier_z_threshold,
    )
    correlations = correlations.sort_values(["target", f"{args.method}_abs_r"], ascending=[True, False])
    needs_feature_correlations = args.feature_correlations or args.group_features or args.combine_representatives
    feature_correlations = (
        feature_feature_correlations(merged, features_to_scan, min_n=args.min_n)
        if needs_feature_correlations
        else pd.DataFrame()
    )
    feature_groups = pd.DataFrame()
    representatives = pd.DataFrame()
    if args.group_features or args.combine_representatives:
        feature_to_group = group_correlated_features(
            features_to_scan,
            feature_correlations,
            method=args.method,
            threshold=args.feature_group_threshold,
        )
        feature_groups, representatives = choose_representative_features(
            features_to_scan,
            feature_to_group,
            correlations,
            method=args.method,
        )
    combined_models = pd.DataFrame()
    combined_predictions = pd.DataFrame()
    if args.combine_representatives:
        combined_models, combined_predictions = evaluate_representative_combinations(
            merged,
            representatives,
            correlations,
            targets,
            method=args.method,
            top_features=args.combo_top_features,
            alpha=args.combo_alpha,
            folds=args.combo_folds,
        )
    covariances = (
        pairwise_covariances(
            merged,
            features_to_scan,
            targets,
            min_n=args.min_n,
            normalization=args.covariance_normalization,
        )
        .sort_values(["target", "abs_covariance"], ascending=[True, False])
        if args.covariance
        else pd.DataFrame()
    )

    device_prefix = f"device{','.join(map(str, devices))}_" if devices else ""
    merged_path = args.output_dir / f"{device_prefix}{args.merged_name}"
    correlation_name = args.correlation_name
    if args.outlier_filter != "none" and correlation_name == DEFAULT_CORRELATION_NAME:
        correlation_name = f"feature_error_correlations_outliers_{args.outlier_filter}.csv"
    correlation_path = args.output_dir / f"{device_prefix}{correlation_name}"
    write_table(merged, merged_path)
    write_table(correlations, correlation_path)
    if needs_feature_correlations:
        feature_correlation_path = args.output_dir / f"{device_prefix}{args.feature_correlation_name}"
        write_table(feature_correlations, feature_correlation_path)
    if args.group_features or args.combine_representatives:
        feature_group_path = args.output_dir / f"{device_prefix}{args.feature_group_name}"
        representative_path = args.output_dir / f"{device_prefix}{args.representative_name}"
        write_table(feature_groups, feature_group_path)
        write_table(representatives, representative_path)
    if args.combine_representatives:
        combined_model_path = args.output_dir / f"{device_prefix}{args.combined_model_name}"
        combined_prediction_path = args.output_dir / f"{device_prefix}{args.combined_prediction_name}"
        combined_scatter_path = args.output_dir / f"{device_prefix}{args.combined_scatter_name}"
        write_table(combined_models, combined_model_path)
        write_table(combined_predictions, combined_prediction_path)
        make_prediction_scatter(combined_predictions, combined_scatter_path)
    if args.covariance:
        covariance_name = args.covariance_name
        if args.covariance_normalization != "none" and covariance_name == DEFAULT_COVARIANCE_NAME:
            covariance_name = f"feature_error_covariances_{args.covariance_normalization}.csv"
        covariance_path = args.output_dir / f"{device_prefix}{covariance_name}"
        write_table(covariances, covariance_path)

    print(f"merged rows        : {len(merged)}")
    print(f"features scanned   : {len(features_to_scan)}")
    print(f"targets            : {', '.join(targets)}")
    if args.outlier_filter != "none":
        print(f"outlier filter     : {args.outlier_filter}")
    print(f"merged csv         : {merged_path}")
    print(f"correlation csv    : {correlation_path}")
    if needs_feature_correlations:
        print(f"feature corr csv   : {feature_correlation_path}")
    if args.group_features or args.combine_representatives:
        print(f"feature groups csv : {feature_group_path}")
        print(f"representative csv : {representative_path}")
        print(f"feature groups     : {len(feature_groups)} at |{args.method} r| >= {args.feature_group_threshold:g}")
    if args.combine_representatives:
        print(f"combined models csv: {combined_model_path}")
        print(f"combined preds csv : {combined_prediction_path}")
        print(f"combined png       : {combined_scatter_path}")
    if args.covariance:
        print(f"covariance csv     : {covariance_path}")
        print(f"covariance norm    : {args.covariance_normalization}")
    for target in targets:
        print(f"\n## {target}")
        print(
            top_correlations(correlations, target=target, method=args.method, n=args.top)[
                ["feature", "n", f"{args.method}_r"]
            ].to_string(index=False)
        )
        if args.covariance:
            print("\nTop covariance")
            print(
                top_covariances(covariances, target=target, n=args.top)[
                    ["feature", "n", "covariance", "feature_std", "target_std"]
                ].to_string(index=False)
            )

    if args.feature_correlations:
        print(f"\n## feature-feature top {args.method}")
        print(
            top_feature_correlations(feature_correlations, method=args.method, n=args.top)[
                ["feature_a", "feature_b", "n", f"{args.method}_r"]
            ].to_string(index=False)
        )
        print(f"\n## feature-feature top {args.method} below |r| <= {args.feature_correlation_max_abs:g}")
        print(
            top_feature_correlations(
                feature_correlations,
                method=args.method,
                n=args.top,
                max_abs_r=args.feature_correlation_max_abs,
            )[["feature_a", "feature_b", "n", f"{args.method}_r"]].to_string(index=False)
        )

    if args.group_features:
        print(f"\n## representative features by group size")
        print(
            feature_groups.head(args.top)[
                ["group_id", "group_size", "representative_feature", "best_target", "best_r"]
            ].to_string(index=False)
        )

    if args.combine_representatives:
        print("\n## combined representative models")
        print(
            combined_models[
                [
                    "target",
                    "selected_feature_count",
                    "best_single_feature",
                    "best_single_r",
                    "cv_r",
                    "fitted_r",
                    "cv_rmse",
                    "cv_mae",
                ]
            ].to_string(index=False)
        )

    if args.representative_scatter:
        if representatives.empty:
            raise ValueError("--representative-scatter requires --group-features.")
        rep_pairs = representative_scatter_pairs(correlations, representatives, method=args.method, n=min(args.top, 18))
        representative_scatter_path = args.output_dir / f"{device_prefix}{args.representative_scatter_name}"
        make_scatter_grid(
            merged,
            rep_pairs,
            representative_scatter_path,
            method=args.method,
            outlier_filter=args.outlier_filter,
            outlier_iqr_multiplier=args.outlier_iqr_multiplier,
            outlier_z_threshold=args.outlier_z_threshold,
        )
        print(f"representative png : {representative_scatter_path}")

    if args.scatter:
        pairs = [
            (row["feature"], row["target"])
            for _, row in top_correlations(correlations, method=args.method, n=min(args.top, 9)).iterrows()
        ]
        scatter_name = "feature_error_scatter.png"
        if args.outlier_filter != "none":
            scatter_name = f"feature_error_scatter_outliers_{args.outlier_filter}.png"
        scatter_path = args.output_dir / f"{device_prefix}{scatter_name}"
        make_scatter_grid(
            merged,
            pairs,
            scatter_path,
            method=args.method,
            outlier_filter=args.outlier_filter,
            outlier_iqr_multiplier=args.outlier_iqr_multiplier,
            outlier_z_threshold=args.outlier_z_threshold,
        )
        print(f"scatter png        : {scatter_path}")


if __name__ == "__main__":
    main()
