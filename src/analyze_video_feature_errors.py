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


def pairwise_correlations(
    df: pd.DataFrame,
    x_cols: Iterable[str],
    y_cols: Iterable[str],
    *,
    methods: Iterable[str] = ("pearson", "spearman"),
    min_n: int = 10,
) -> pd.DataFrame:
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
            row: dict[str, object] = {"target": y_col, "feature": x_col, "n": len(xy)}
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


def make_scatter_grid(
    df: pd.DataFrame,
    pairs: Iterable[tuple[str, str]],
    output_path: Path,
    *,
    method: str = "pearson",
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
    parser.add_argument("--device", nargs="+", help="One or more device ids, e.g. --device 3 or --device 3,5")
    parser.add_argument("--model", default="linear")
    parser.add_argument("--targets", nargs="+", help="Error target columns to analyze. Defaults to common abs-error columns.")
    parser.add_argument("--feature-regex", help="Only analyze feature columns matching this regex.")
    parser.add_argument("--include-meta", action="store_true", help="Include mostly static metadata columns in correlation scans.")
    parser.add_argument("--min-n", type=int, default=10)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--method", choices=["pearson", "spearman"], default="pearson")
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
    correlations = pairwise_correlations(merged, features_to_scan, targets, min_n=args.min_n)
    correlations = correlations.sort_values(["target", f"{args.method}_abs_r"], ascending=[True, False])

    device_prefix = f"device{','.join(map(str, devices))}_" if devices else ""
    merged_path = args.output_dir / f"{device_prefix}{args.merged_name}"
    correlation_path = args.output_dir / f"{device_prefix}{args.correlation_name}"
    write_table(merged, merged_path)
    write_table(correlations, correlation_path)

    print(f"merged rows        : {len(merged)}")
    print(f"features scanned   : {len(features_to_scan)}")
    print(f"targets            : {', '.join(targets)}")
    print(f"merged csv         : {merged_path}")
    print(f"correlation csv    : {correlation_path}")
    for target in targets:
        print(f"\n## {target}")
        print(
            top_correlations(correlations, target=target, method=args.method, n=args.top)[
                ["feature", "n", f"{args.method}_r"]
            ].to_string(index=False)
        )

    if args.scatter:
        pairs = [
            (row["feature"], row["target"])
            for _, row in top_correlations(correlations, method=args.method, n=min(args.top, 9)).iterrows()
        ]
        scatter_path = args.output_dir / f"{device_prefix}feature_error_scatter.png"
        make_scatter_grid(merged, pairs, scatter_path, method=args.method)
        print(f"scatter png        : {scatter_path}")


if __name__ == "__main__":
    main()
