import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from draw_graph import (
    INPUT_PATH,
    calculate_linear_regression,
    clean_regression_data,
    format_float,
)


COMPARISON_OUTPUT_PATH = Path("validation/regression_model_comparison.csv")
EXP_FLUX_SCALE = 1_000_000.0
FLAT_EXP_MIN_LOG_SLOPE = 0.001
FLAT_EXP_MAX_LOG_SLOPE = 0.05
FLAT_EXP_GRID_SIZE = 600

COMPARISON_JOBS = [
    {
        "figure_name": "in count model comparison",
        "title": "In count vs filtered in flux",
        "x_col": "total_filtered_in_flux",
        "y_col": "in",
        "linear_color": "black",
        "exp_color": "crimson",
        "output_path": Path("validation/in_count_model_comparison.png"),
    },
    {
        "figure_name": "out count model comparison",
        "title": "Out count vs filtered out flux",
        "x_col": "total_filtered_out_flux",
        "y_col": "out",
        "linear_color": "black",
        "exp_color": "crimson",
        "output_path": Path("validation/out_count_model_comparison.png"),
    },
]


def calculate_prediction_metrics(y, y_pred, num_params):
    residuals = y - y_pred
    n = len(y)
    ss_res = float(np.sum(residuals**2))
    ss_yy = float(np.sum((y - np.mean(y)) ** 2))
    mse = ss_res / n
    df_resid = n - num_params

    metrics = {
        "r_squared": np.nan,
        "adjusted_r_squared": np.nan,
        "mae": float(np.mean(np.abs(residuals))),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "residual_std_error": np.nan,
    }

    if ss_yy > 0:
        metrics["r_squared"] = 1 - (ss_res / ss_yy)
        if df_resid > 0:
            metrics["adjusted_r_squared"] = 1 - (
                (1 - metrics["r_squared"]) * (n - 1) / df_resid
            )

    if df_resid > 0:
        metrics["residual_std_error"] = math.sqrt(ss_res / df_resid)

    return metrics


def predict_linear(result, x):
    return result["intercept"] + result["slope"] * x


def predict_flat_exponential(result, x):
    scaled_x = x / result["x_scale"]
    curve = np.expm1(result["log_slope"] * scaled_x)
    return result["intercept"] + result["a"] * curve


def calculate_flat_exponential_regression(
    data,
    x_col,
    y_col,
    label,
    x_scale=EXP_FLUX_SCALE,
):
    x = data[x_col].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)
    n = len(data)

    result = {
        "label": label,
        "model": "flat_exponential",
        "x_col": x_col,
        "y_col": y_col,
        "status": "ok",
        "n": n,
        "fit_n": n,
        "slope": np.nan,
        "intercept": np.nan,
        "x_scale": x_scale,
        "log_slope": np.nan,
        "a": np.nan,
        "b": np.nan,
        "r": np.nan,
        "r_squared": np.nan,
        "adjusted_r_squared": np.nan,
        "mae": np.nan,
        "mse": np.nan,
        "rmse": np.nan,
        "residual_std_error": np.nan,
        "x_mean": np.nan,
        "y_mean": np.nan,
        "equation": "",
    }

    if n < 3:
        result["status"] = "insufficient_data"
        return result

    scaled_x = x / x_scale
    best = None
    search_low = FLAT_EXP_MIN_LOG_SLOPE
    search_high = FLAT_EXP_MAX_LOG_SLOPE

    for _ in range(4):
        for log_slope in np.linspace(search_low, search_high, FLAT_EXP_GRID_SIZE):
            curve = np.expm1(log_slope * scaled_x)
            design = np.column_stack([curve, np.ones_like(curve)])
            amplitude, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
            if amplitude < 0:
                continue

            y_pred = intercept + amplitude * curve
            sse = float(np.sum((y - y_pred) ** 2))
            if best is None or sse < best["sse"]:
                best = {
                    "sse": sse,
                    "log_slope": float(log_slope),
                    "amplitude": float(amplitude),
                    "intercept": float(intercept),
                    "curve": curve,
                    "y_pred": y_pred,
                }

        if best is None:
            break

        step = (search_high - search_low) / max(FLAT_EXP_GRID_SIZE - 1, 1)
        search_low = max(FLAT_EXP_MIN_LOG_SLOPE, best["log_slope"] - step)
        search_high = min(FLAT_EXP_MAX_LOG_SLOPE, best["log_slope"] + step)

    if best is None:
        result["status"] = "no_positive_amplitude_fit"
        return result

    log_slope = best["log_slope"]
    amplitude = best["amplitude"]
    intercept = best["intercept"]
    curve = best["curve"]
    y_pred = best["y_pred"]

    result.update(
        {
            "slope": amplitude * log_slope / x_scale,
            "intercept": intercept,
            "log_slope": log_slope,
            "a": amplitude,
            "b": log_slope / x_scale,
            "x_mean": float(np.mean(x)),
            "y_mean": float(np.mean(y)),
            "equation": (
                f"{y_col} = {intercept:.10g} + {amplitude:.10g} * "
                f"(exp({log_slope:.10g} * ({x_col} / {x_scale:.10g})) - 1)"
            ),
        }
    )
    result.update(calculate_prediction_metrics(y, y_pred, num_params=3))

    curve_centered = curve - float(np.mean(curve))
    y_centered = y - float(np.mean(y))
    ss_curve = float(np.sum(curve_centered**2))
    ss_y = float(np.sum(y_centered**2))
    ss_curve_y = float(np.sum(curve_centered * y_centered))
    if ss_curve > 0 and ss_y > 0:
        result["r"] = ss_curve_y / math.sqrt(ss_curve * ss_y)

    return result


def linear_result_for_comparison(linear_result):
    result = dict(linear_result)
    result["model"] = "linear"
    result["fit_n"] = result["n"]
    result["x_scale"] = 1.0
    result["log_slope"] = np.nan
    result["a"] = np.nan
    result["b"] = np.nan
    return result


def draw_model_comparison_plot(df, job, clean_data, linear_result, exp_result):
    plt.figure(job["figure_name"], figsize=(12, 6))

    scatter_kwargs = {
        "data": df,
        "x": job["x_col"],
        "y": job["y_col"],
    }
    if "time" in df.columns:
        scatter_kwargs.update({"hue": "time", "palette": "tab20"})

    sns.scatterplot(**scatter_kwargs)

    x_values = clean_data[job["x_col"]].to_numpy(dtype=float)
    y_values = clean_data[job["y_col"]].to_numpy(dtype=float)
    x_line = np.linspace(float(np.min(x_values)), float(np.max(x_values)), 200)

    if linear_result["status"] == "ok":
        plt.plot(
            x_line,
            predict_linear(linear_result, x_line),
            color=job["linear_color"],
            linewidth=2,
            label=(
                "linear "
                f"(R^2={format_float(linear_result['r_squared'])}, "
                f"RMSE={format_float(linear_result['rmse'])})"
            ),
        )

    if exp_result["status"] == "ok":
        plt.plot(
            x_line,
            predict_flat_exponential(exp_result, x_line),
            color=job["exp_color"],
            linewidth=2,
            linestyle="--",
            label=(
                "flat exponential "
                f"(R^2={format_float(exp_result['r_squared'])}, "
                f"RMSE={format_float(exp_result['rmse'])})"
            ),
        )

    subtitle = (
        f"linear: R^2={format_float(linear_result['r_squared'])}, "
        f"RMSE={format_float(linear_result['rmse'])} / "
        f"exp: R^2={format_float(exp_result['r_squared'])}, "
        f"RMSE={format_float(exp_result['rmse'])}"
    )
    plt.title(f"{job['title']}\n{subtitle}")

    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc="upper left")

    y_min = min(0, float(np.min(y_values)))
    y_max = float(np.max(y_values))
    if y_max > y_min:
        plt.ylim(y_min, y_max * 1.15)

    plt.tight_layout()
    plt.grid(True)
    plt.savefig(job["output_path"], dpi=180, bbox_inches="tight")


def print_model_comparison(stats_df):
    display_cols = [
        "label",
        "model",
        "status",
        "n",
        "r_squared",
        "adjusted_r_squared",
        "mae",
        "rmse",
    ]
    print("\nRegression model comparison")
    print(stats_df[display_cols].to_string(index=False))


def main():
    df = pd.read_excel(INPUT_PATH)
    if "time" in df.columns:
        df["time"] = df["time"].astype(str)

    runs = []
    rows = []

    for job in COMPARISON_JOBS:
        clean_data = clean_regression_data(df, job["x_col"], job["y_col"])
        linear_result = linear_result_for_comparison(
            calculate_linear_regression(
                clean_data,
                job["x_col"],
                job["y_col"],
                job["figure_name"],
            )
        )
        exp_result = calculate_flat_exponential_regression(
            clean_data,
            job["x_col"],
            job["y_col"],
            job["figure_name"],
        )

        rows.extend([linear_result, exp_result])
        runs.append((job, clean_data, linear_result, exp_result))

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(COMPARISON_OUTPUT_PATH, index=False)
    print_model_comparison(stats_df)
    print(f"\nSaved model comparison to: {COMPARISON_OUTPUT_PATH}")

    for index, (job, clean_data, linear_result, exp_result) in enumerate(runs):
        draw_model_comparison_plot(df, job, clean_data, linear_result, exp_result)
        print(f"Saved plot to: {job['output_path']}")
        plt.show(block=index == len(runs) - 1)


if __name__ == "__main__":
    main()
