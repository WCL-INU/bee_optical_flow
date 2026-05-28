import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

try:
    from src.bee_entrance_count import Config, compare_videos, process_video
except ModuleNotFoundError:
    from bee_entrance_count import Config, compare_videos, process_video


VIDEO_DIR = Path("videos")
OUTPUT_ROOT = Path("bee_count_output") / "runs"
DEFAULT_PATTERN = "ANU-25-summer-6_*.mp4"


PRESETS = {
    "raw": Config(
        blur_kernel=3,
        use_persistence_filter=False,
        preview_stride=3,
    ),
    "persistence": Config(
        blur_kernel=3,
        preview_stride=3,
    ),
    "blur5": Config(
        blur_kernel=5,
        preview_stride=3,
    ),
    "strict_noise": Config(
        blur_kernel=5,
        flow_mag_threshold=1.5,
        normal_flow_threshold=0.8,
        use_component_area_filter=True,
        min_flow_component_area=400,
        preview_stride=3,
    ),
    "selected": Config(
        blur_kernel=5,
        flow_mag_threshold=1.0,
        normal_flow_threshold=0.5,
        use_component_area_filter=True,
        min_flow_component_area=200,
        preview_stride=3,
    ),
}

TUNE_GRIDS = {
    "quick": [
        {
            "name": "raw_blur3",
            "blur_kernel": 3,
            "flow_mag_threshold": 0.30,
            "normal_flow_threshold": 0.08,
            "use_persistence_filter": False,
            "use_component_area_filter": False,
            "min_flow_component_area": 30,
        },
        {
            "name": "low_raw_blur1",
            "blur_kernel": 1,
            "flow_mag_threshold": 0.10,
            "normal_flow_threshold": 0.02,
            "use_persistence_filter": False,
            "use_component_area_filter": False,
            "min_flow_component_area": 30,
        },
        {
            "name": "low_persist_blur1",
            "blur_kernel": 1,
            "flow_mag_threshold": 0.10,
            "normal_flow_threshold": 0.02,
            "use_persistence_filter": True,
            "persist_decay": 0.65,
            "persist_threshold": 1.3,
            "use_component_area_filter": False,
            "min_flow_component_area": 30,
        },
        {
            "name": "low_persist_blur3",
            "blur_kernel": 3,
            "flow_mag_threshold": 0.10,
            "normal_flow_threshold": 0.02,
            "use_persistence_filter": True,
            "persist_decay": 0.65,
            "persist_threshold": 1.3,
            "use_component_area_filter": False,
            "min_flow_component_area": 30,
        },
        {
            "name": "low_persist_area100",
            "blur_kernel": 3,
            "flow_mag_threshold": 0.10,
            "normal_flow_threshold": 0.02,
            "use_persistence_filter": True,
            "persist_decay": 0.65,
            "persist_threshold": 1.3,
            "use_component_area_filter": True,
            "min_flow_component_area": 100,
        },
        {
            "name": "persistence_blur3",
            "blur_kernel": 3,
            "flow_mag_threshold": 0.30,
            "normal_flow_threshold": 0.08,
            "use_persistence_filter": True,
            "persist_decay": 0.65,
            "persist_threshold": 1.3,
            "use_component_area_filter": False,
            "min_flow_component_area": 30,
        },
        {
            "name": "blur5_no_persist",
            "blur_kernel": 5,
            "flow_mag_threshold": 0.30,
            "normal_flow_threshold": 0.08,
            "use_persistence_filter": False,
            "use_component_area_filter": False,
            "min_flow_component_area": 30,
        },
        {
            "name": "selected_area400",
            "blur_kernel": 5,
            "flow_mag_threshold": 1.5,
            "normal_flow_threshold": 0.8,
            "use_component_area_filter": True,
            "min_flow_component_area": 400,
        },
        {
            "name": "selected_area200",
            "blur_kernel": 5,
            "flow_mag_threshold": 1.5,
            "normal_flow_threshold": 0.8,
            "use_component_area_filter": True,
            "min_flow_component_area": 200,
        },
        {
            "name": "selected_no_area",
            "blur_kernel": 5,
            "flow_mag_threshold": 1.5,
            "normal_flow_threshold": 0.8,
            "use_component_area_filter": False,
            "min_flow_component_area": 30,
        },
        {
            "name": "soft_area200",
            "blur_kernel": 5,
            "flow_mag_threshold": 1.0,
            "normal_flow_threshold": 0.5,
            "use_component_area_filter": True,
            "min_flow_component_area": 200,
        },
        {
            "name": "soft_no_area",
            "blur_kernel": 5,
            "flow_mag_threshold": 1.0,
            "normal_flow_threshold": 0.5,
            "use_component_area_filter": False,
            "min_flow_component_area": 30,
        },
        {
            "name": "default_thresholds",
            "blur_kernel": 5,
            "flow_mag_threshold": 0.30,
            "normal_flow_threshold": 0.08,
            "use_component_area_filter": False,
            "min_flow_component_area": 30,
        },
    ],
}


def discover_videos(video_dir, pattern):
    video_dir = Path(video_dir)
    return sorted(video_dir.glob(pattern))


def parse_video_list(paths):
    return [Path(path) for path in paths]


def select_videos(args):
    if args.videos:
        videos = parse_video_list(args.videos)
    else:
        videos = discover_videos(args.video_dir, args.pattern)

    if args.start is not None or args.end is not None:
        start = args.start or 0
        end = args.end
        videos = videos[start:end]

    if args.limit is not None:
        videos = videos[: args.limit]

    if not videos:
        raise RuntimeError("No videos selected. Check --video-dir, --pattern, or --videos.")

    missing = [video for video in videos if not video.exists()]
    if missing:
        missing_text = "\n".join(str(video) for video in missing)
        raise FileNotFoundError(f"Selected video files do not exist:\n{missing_text}")

    return videos


def build_config(args):
    config = PRESETS[args.preset]

    updates = {}
    for field_name in [
        "blur_kernel",
        "flow_mag_threshold",
        "normal_flow_threshold",
        "persist_decay",
        "persist_threshold",
        "min_flow_component_area",
        "preview_stride",
        "warn_ratio_between_videos",
        "warn_max_window_filtered_traffic_count_est",
        "balance_ratio_threshold",
    ]:
        value = getattr(args, field_name)
        if value is not None:
            updates[field_name] = value

    if args.no_persistence_filter:
        updates["use_persistence_filter"] = False
    if args.use_component_area_filter:
        updates["use_component_area_filter"] = True
    if args.no_component_area_filter:
        updates["use_component_area_filter"] = False
    if args.use_global_flow_compensation:
        updates["use_global_flow_compensation"] = True
    if args.use_bidirectional_balance_filter:
        updates["use_bidirectional_balance_filter"] = True

    return replace(config, **updates)


def config_from_overrides(base_config, overrides):
    updates = {key: value for key, value in overrides.items() if key != "name"}
    return replace(base_config, **updates)


def load_truth(truth_csv):
    truth = pd.read_csv(truth_csv)
    truth.columns = [column.strip() for column in truth.columns]
    required = {"video", "in", "out"}
    missing = required - set(truth.columns)
    if missing:
        raise ValueError(f"Truth CSV is missing columns: {sorted(missing)}")
    truth["video"] = truth["video"].astype(str).str.strip()
    truth["in"] = pd.to_numeric(truth["in"])
    truth["out"] = pd.to_numeric(truth["out"])
    truth["true_traffic"] = truth["in"] + truth["out"]
    return truth


def fit_flux_units(df):
    in_positive = df["in"] > 0
    out_positive = df["out"] > 0
    in_unit = (
        df.loc[in_positive, "total_filtered_in_flux"].sum()
        / max(df.loc[in_positive, "in"].sum(), 1e-6)
    )
    out_unit = (
        df.loc[out_positive, "total_filtered_out_flux"].sum()
        / max(df.loc[out_positive, "out"].sum(), 1e-6)
    )
    return max(float(in_unit), 1e-6), max(float(out_unit), 1e-6)


def evaluate_summary(
    summary_df,
    truth_df,
    output_dir,
    run_name="run",
    target_video=None,
    zero_weight=2.0,
    target_weight=1.5,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = summary_df.merge(truth_df, on="video", how="left", validate="one_to_one")
    if df[["in", "out"]].isna().any(axis=None):
        missing = df.loc[df[["in", "out"]].isna().any(axis=1), "video"].tolist()
        raise ValueError(f"Missing truth rows for videos: {missing}")

    in_unit, out_unit = fit_flux_units(df)
    df["pred_in"] = df["total_filtered_in_flux"] / in_unit
    df["pred_out"] = df["total_filtered_out_flux"] / out_unit
    df["pred_traffic"] = df["pred_in"] + df["pred_out"]
    df["in_abs_error"] = (df["pred_in"] - df["in"]).abs()
    df["out_abs_error"] = (df["pred_out"] - df["out"]).abs()
    df["traffic_abs_error"] = (df["pred_traffic"] - df["true_traffic"]).abs()
    df["traffic_signed_error"] = df["pred_traffic"] - df["true_traffic"]
    df["traffic_abs_pct_error"] = df["traffic_abs_error"] / df["true_traffic"].clip(lower=1.0)

    zero_df = df[df["true_traffic"] == 0]
    active_df = df[df["true_traffic"] > 0]
    target_row = (
        df[df["video"] == target_video].iloc[0]
        if target_video and (df["video"] == target_video).any()
        else None
    )

    metrics = {
        "run": run_name,
        "video_count": len(df),
        "in_flux_unit": in_unit,
        "out_flux_unit": out_unit,
        "traffic_mae": float(df["traffic_abs_error"].mean()),
        "active_traffic_mae": float(active_df["traffic_abs_error"].mean())
        if not active_df.empty
        else 0.0,
        "zero_mean_pred_traffic": float(zero_df["pred_traffic"].mean())
        if not zero_df.empty
        else 0.0,
        "zero_max_pred_traffic": float(zero_df["pred_traffic"].max())
        if not zero_df.empty
        else 0.0,
        "mean_abs_pct_error": float(df["traffic_abs_pct_error"].mean()),
        "target_video": target_video or "",
        "target_true_traffic": float(target_row["true_traffic"]) if target_row is not None else 0.0,
        "target_pred_traffic": float(target_row["pred_traffic"]) if target_row is not None else 0.0,
        "target_abs_error": float(target_row["traffic_abs_error"]) if target_row is not None else 0.0,
        "zero_weight": zero_weight,
        "target_weight": target_weight,
    }
    metrics["score"] = (
        metrics["active_traffic_mae"]
        + zero_weight * metrics["zero_mean_pred_traffic"]
        + target_weight * metrics["target_abs_error"]
    )

    eval_path = output_dir / "evaluation.csv"
    metrics_path = output_dir / "evaluation_metrics.csv"
    df.to_csv(eval_path, index=False)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    return df, metrics


def make_run_dir(args):
    run_name = args.run_name or args.preset
    return Path(args.output_root) / run_name


def print_plan(videos, config, output_dir, mode):
    print(f"mode       : {mode}")
    print(f"output dir : {output_dir}")
    print(f"videos     : {len(videos)}")
    for idx, video in enumerate(videos, start=1):
        print(f"  {idx:02d}. {video}")
    print(
        "config     : "
        f"blur={config.blur_kernel}, "
        f"mag>{config.flow_mag_threshold}, "
        f"normal>{config.normal_flow_threshold}, "
        f"persistence={config.use_persistence_filter}, "
        f"decay={config.persist_decay}, "
        f"threshold={config.persist_threshold}, "
        f"area_filter={config.use_component_area_filter}, "
        f"min_area={config.min_flow_component_area}, "
        f"preview_stride={config.preview_stride}"
    )


def run_batch(videos, output_dir, config):
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for idx, video in enumerate(videos, start=1):
        print(f"[{idx}/{len(videos)}] Processing {video}")
        result, _, _ = process_video(video, output_dir, config)
        summaries.append(result)

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "batch_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"batch summary: {summary_path}")
    return summary_df


def evaluate_existing_summary(summary_csv, truth_csv, output_dir, target_video, args):
    summary_df = pd.read_csv(summary_csv)
    truth_df = load_truth(truth_csv)
    eval_df, metrics = evaluate_summary(
        summary_df,
        truth_df,
        output_dir,
        run_name=Path(summary_csv).parent.name,
        target_video=target_video,
        zero_weight=args.zero_weight,
        target_weight=args.target_weight,
    )
    print(f"evaluation: {Path(output_dir) / 'evaluation.csv'}")
    print(f"metrics   : {Path(output_dir) / 'evaluation_metrics.csv'}")
    print(
        "calibrated units: "
        f"IN={metrics['in_flux_unit']:.2f}, OUT={metrics['out_flux_unit']:.2f}"
    )
    print(
        "errors: "
        f"traffic_mae={metrics['traffic_mae']:.2f}, "
        f"target_abs_error={metrics['target_abs_error']:.2f}, "
        f"zero_mean_pred={metrics['zero_mean_pred_traffic']:.2f}"
    )
    if target_video:
        target = eval_df[eval_df["video"] == target_video]
        if not target.empty:
            row = target.iloc[0]
            print(
                f"target {target_video}: true={row['true_traffic']:.1f}, "
                f"pred={row['pred_traffic']:.1f}"
            )
    return eval_df, metrics


def run_tune(videos, output_dir, base_config, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    truth_df = load_truth(args.truth_csv)
    trials = TUNE_GRIDS[args.tune_grid]
    metrics_rows = []

    for idx, trial in enumerate(trials, start=1):
        trial_name = trial["name"]
        trial_dir = output_dir / trial_name
        trial_config = config_from_overrides(
            replace(base_config, preview_stride=args.tune_preview_stride),
            trial,
        )
        summary_path = trial_dir / "batch_summary.csv"

        print(f"[tune {idx}/{len(trials)}] {trial_name}")
        if args.reuse_existing and summary_path.exists():
            print(f"reuse: {summary_path}")
            summary_df = pd.read_csv(summary_path)
        else:
            summary_df = run_batch(videos, trial_dir, trial_config)

        _, metrics = evaluate_summary(
            summary_df,
            truth_df,
            trial_dir,
            run_name=trial_name,
            target_video=args.target_video,
            zero_weight=args.zero_weight,
            target_weight=args.target_weight,
        )
        metrics.update(
            {
                "blur_kernel": trial_config.blur_kernel,
                "flow_mag_threshold": trial_config.flow_mag_threshold,
                "normal_flow_threshold": trial_config.normal_flow_threshold,
                "use_component_area_filter": trial_config.use_component_area_filter,
                "min_flow_component_area": trial_config.min_flow_component_area,
            }
        )
        metrics_rows.append(metrics)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("score")
    tuning_path = output_dir / "tuning_results.csv"
    metrics_df.to_csv(tuning_path, index=False)
    print(f"tuning results: {tuning_path}")
    print(
        metrics_df[
            [
                "run",
                "score",
                "active_traffic_mae",
                "zero_mean_pred_traffic",
                "target_pred_traffic",
                "target_abs_error",
                "blur_kernel",
                "flow_mag_threshold",
                "normal_flow_threshold",
                "use_component_area_filter",
                "min_flow_component_area",
            ]
        ].to_string(index=False)
    )
    return metrics_df


def run_groups(videos, output_dir, config, group_size, slide):
    if group_size < 2:
        raise ValueError("--group-size must be at least 2 in groups mode.")
    if slide < 1:
        raise ValueError("--slide must be at least 1.")

    group_dirs = []
    rows = []
    last_start = max(0, len(videos) - group_size)
    starts = range(0, last_start + 1, slide)

    for group_idx, start in enumerate(starts, start=1):
        group = videos[start : start + group_size]
        if len(group) < group_size:
            continue
        group_dir = output_dir / f"group_{group_idx:03d}_{group[0].stem}_to_{group[-1].stem}"
        group_dirs.append(group_dir)
        print(f"[group {group_idx}] {group[0].name} -> {group[-1].name}")
        summary_df, warnings = compare_videos(group, group_dir, config)
        group_summary = summary_df.copy()
        group_summary.insert(0, "group", group_idx)
        group_summary.insert(1, "warning_count", len(warnings))
        rows.append(group_summary)

    if rows:
        all_groups_df = pd.concat(rows, ignore_index=True)
    else:
        all_groups_df = pd.DataFrame()
    summary_path = output_dir / "group_summary.csv"
    all_groups_df.to_csv(summary_path, index=False)
    print(f"group summary: {summary_path}")
    return group_dirs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convenient runner for bee entrance optical-flow experiments."
    )
    parser.add_argument(
        "--mode",
        choices=["batch", "compare", "groups", "evaluate", "tune"],
        default="batch",
        help=(
            "batch: process each video; compare: one comparison; groups: sliding "
            "comparisons; evaluate: compare a summary CSV to truth; tune: run a "
            "small parameter grid and evaluate truth."
        ),
    )
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--videos", nargs="+", type=Path)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--truth-csv", type=Path, default=Path("videos") / "entrance.csv")
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument(
        "--target-video",
        default="ANU-25-summer-6_20260405_160000.mp4",
        help="Target video to highlight in truth-based evaluation.",
    )
    parser.add_argument("--zero-weight", type=float, default=2.0)
    parser.add_argument("--target-weight", type=float, default=1.5)

    parser.add_argument("--preset", choices=sorted(PRESETS), default="selected")
    parser.add_argument("--blur-kernel", type=int, choices=[1, 3, 5])
    parser.add_argument("--flow-mag-threshold", type=float)
    parser.add_argument("--normal-flow-threshold", type=float)
    parser.add_argument("--no-persistence-filter", action="store_true")
    parser.add_argument("--persist-decay", type=float)
    parser.add_argument("--persist-threshold", type=float)
    parser.add_argument("--use-component-area-filter", action="store_true")
    parser.add_argument("--no-component-area-filter", action="store_true")
    parser.add_argument("--min-flow-component-area", type=int)
    parser.add_argument("--use-global-flow-compensation", action="store_true")
    parser.add_argument("--use-bidirectional-balance-filter", action="store_true")
    parser.add_argument("--balance-ratio-threshold", type=float)
    parser.add_argument("--preview-stride", type=int)
    parser.add_argument("--warn-ratio-between-videos", type=float)
    parser.add_argument("--warn-max-window-filtered-traffic-count-est", type=float)

    parser.add_argument("--group-size", type=int, default=3)
    parser.add_argument("--slide", type=int, default=1)
    parser.add_argument("--tune-grid", choices=sorted(TUNE_GRIDS), default="quick")
    parser.add_argument("--tune-preview-stride", type=int, default=999999)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = build_config(args)
    output_dir = make_run_dir(args)

    if args.mode == "evaluate":
        if args.summary_csv is None:
            args.summary_csv = output_dir / "batch_summary.csv"
        print(f"mode       : evaluate")
        print(f"summary csv: {args.summary_csv}")
        print(f"truth csv  : {args.truth_csv}")
        print(f"output dir : {output_dir}")
        if args.dry_run:
            print("dry run: no evaluation performed")
            return
        evaluate_existing_summary(
            args.summary_csv,
            args.truth_csv,
            output_dir,
            args.target_video,
            args,
        )
        return

    videos = select_videos(args)
    print_plan(videos, config, output_dir, args.mode)
    if args.dry_run:
        print("dry run: no videos processed")
        return

    if args.mode == "batch":
        run_batch(videos, output_dir, config)
    elif args.mode == "compare":
        compare_videos(videos, output_dir, config)
    elif args.mode == "tune":
        run_tune(videos, output_dir, config, args)
    else:
        run_groups(videos, output_dir, config, args.group_size, args.slide)


if __name__ == "__main__":
    main()
