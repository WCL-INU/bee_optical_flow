import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


INPUT_PATH = Path("bee_count_output/Book1.xlsx")
STATS_OUTPUT_PATH = Path("bee_count_output/linear_regression_stats.csv")

REGRESSION_JOBS = [
    {
        "figure_name": "in count",
        "title": "In count vs filtered in flux",
        "x_col": "total_filtered_in_flux",
        "y_col": "in",
        "line_color": "black",
    },
    {
        "figure_name": "out count",
        "title": "Out count vs filtered out flux",
        "x_col": "total_filtered_out_flux",
        "y_col": "out",
        "line_color": "black",
    },
]


def beta_continued_fraction(a, b, x):
    max_iterations = 200
    epsilon = 3e-14
    tiny = 1e-300

    qab = a + b
    qap = a + 1
    qam = a - 1
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d

    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < epsilon:
            break

    return h


def regularized_incomplete_beta(x, a, b):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    log_beta = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    beta_term = math.exp(log_beta)

    if x < (a + 1.0) / (a + b + 2.0):
        return beta_term * beta_continued_fraction(a, b, x) / a

    return 1.0 - beta_term * beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_survival(t_value, df):
    if df <= 0:
        return np.nan
    if math.isinf(t_value):
        return 0.0

    t_value = abs(float(t_value))
    x = df / (df + t_value**2)
    return 0.5 * regularized_incomplete_beta(x, df / 2.0, 0.5)


def student_t_two_tailed_p_value(t_value, df):
    p_value = 2.0 * student_t_survival(t_value, df)
    return min(max(p_value, 0.0), 1.0)


def student_t_critical_value(df, confidence=0.95):
    target_survival = (1.0 - confidence) / 2.0
    low = 0.0
    high = 1.0

    while student_t_survival(high, df) > target_survival:
        high *= 2.0

    for _ in range(80):
        midpoint = (low + high) / 2.0
        if student_t_survival(midpoint, df) > target_survival:
            low = midpoint
        else:
            high = midpoint

    return (low + high) / 2.0


def clean_regression_data(df, x_col, y_col):
    """회귀에 사용할 두 컬럼을 숫자로 변환하고 결측/무한값을 제거한다."""
    data = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    return data


def calculate_linear_regression(data, x_col, y_col, label):
    x = data[x_col].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)
    n = len(data)

    result = {
        "label": label,
        "x_col": x_col,
        "y_col": y_col,
        "status": "ok",
        "n": n,
        "slope": np.nan,
        "intercept": np.nan,
        "r": np.nan,
        "r_squared": np.nan,
        "adjusted_r_squared": np.nan,
        "mae": np.nan,
        "mse": np.nan,
        "rmse": np.nan,
        "residual_std_error": np.nan,
        "slope_std_error": np.nan,
        "intercept_std_error": np.nan,
        "t_stat": np.nan,
        "p_value": np.nan,
        "slope_ci95_low": np.nan,
        "slope_ci95_high": np.nan,
        "intercept_ci95_low": np.nan,
        "intercept_ci95_high": np.nan,
        "x_mean": np.nan,
        "y_mean": np.nan,
        "equation": "",
    }

    if n < 2:
        result["status"] = "insufficient_data"
        return result

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_centered = x - x_mean
    y_centered = y - y_mean
    ss_xx = float(np.sum(x_centered**2))
    ss_yy = float(np.sum(y_centered**2))

    result["x_mean"] = x_mean
    result["y_mean"] = y_mean

    if ss_xx == 0:
        result["status"] = "constant_x"
        return result

    ss_xy = float(np.sum(x_centered * y_centered))
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    y_pred = intercept + slope * x
    residuals = y - y_pred
    ss_res = float(np.sum(residuals**2))
    mse = ss_res / n
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(residuals)))
    df_resid = n - 2

    result.update(
        {
            "slope": slope,
            "intercept": intercept,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "equation": f"{y_col} = {slope:.10g} * {x_col} + {intercept:.10g}",
        }
    )

    if ss_yy > 0:
        result["r"] = ss_xy / math.sqrt(ss_xx * ss_yy)
        result["r_squared"] = 1 - (ss_res / ss_yy)
        if n > 2:
            result["adjusted_r_squared"] = 1 - (
                (1 - result["r_squared"]) * (n - 1) / (n - 2)
            )

    if df_resid <= 0:
        return result

    residual_std_error = math.sqrt(ss_res / df_resid)
    slope_std_error = residual_std_error / math.sqrt(ss_xx)
    intercept_std_error = residual_std_error * math.sqrt((1 / n) + (x_mean**2 / ss_xx))

    result.update(
        {
            "residual_std_error": residual_std_error,
            "slope_std_error": slope_std_error,
            "intercept_std_error": intercept_std_error,
        }
    )

    if slope_std_error == 0:
        result["t_stat"] = math.inf if slope != 0 else np.nan
    else:
        result["t_stat"] = slope / slope_std_error

    if np.isfinite(result["t_stat"]):
        t_critical = student_t_critical_value(df_resid)
        result["p_value"] = student_t_two_tailed_p_value(result["t_stat"], df_resid)
        result["slope_ci95_low"] = slope - t_critical * slope_std_error
        result["slope_ci95_high"] = slope + t_critical * slope_std_error
        result["intercept_ci95_low"] = intercept - t_critical * intercept_std_error
        result["intercept_ci95_high"] = intercept + t_critical * intercept_std_error
    elif math.isinf(result["t_stat"]):
        result["p_value"] = 0.0
        result["slope_ci95_low"] = slope
        result["slope_ci95_high"] = slope
        result["intercept_ci95_low"] = intercept
        result["intercept_ci95_high"] = intercept

    return result


def format_float(value, digits=4):
    if pd.isna(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    return f"{value:.{digits}g}"


def draw_relationship_plot(df, job, regression_result, clean_data):
    plt.figure(job["figure_name"], figsize=(12, 6))

    scatter_kwargs = {
        "data": df,
        "x": job["x_col"],
        "y": job["y_col"],
    }
    if "time" in df.columns:
        scatter_kwargs.update({"hue": "time", "palette": "tab20"})

    sns.scatterplot(**scatter_kwargs)

    if regression_result["status"] == "ok":
        x_values = clean_data[job["x_col"]].to_numpy(dtype=float)
        x_line = np.linspace(float(np.min(x_values)), float(np.max(x_values)), 100)
        y_line = regression_result["intercept"] + regression_result["slope"] * x_line

        plt.plot(
            x_line,
            y_line,
            color=job["line_color"],
            linewidth=2,
            label="linear regression",
        )

        subtitle = (
            f"y = {format_float(regression_result['slope'])}x "
            f"+ {format_float(regression_result['intercept'])}, "
            f"R^2 = {format_float(regression_result['r_squared'])}, "
            f"n = {regression_result['n']}"
        )
    else:
        subtitle = f"Regression skipped: {regression_result['status']}"

    plt.title(f"{job['title']}\n{subtitle}")
    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.grid(True)


def print_regression_summary(stats_df):
    display_cols = [
        "label",
        "status",
        "n",
        "slope",
        "intercept",
        "r",
        "r_squared",
        "adjusted_r_squared",
        "mae",
        "rmse",
        "p_value",
    ]

    print("\nLinear regression summary")
    print(stats_df[display_cols].to_string(index=False))


def main():
    # 1. 엑셀 데이터 불러오기
    df = pd.read_excel(INPUT_PATH)

    if "time" in df.columns:
        df["time"] = df["time"].astype(str)

    regression_runs = []

    for job in REGRESSION_JOBS:
        clean_data = clean_regression_data(df, job["x_col"], job["y_col"])
        regression_result = calculate_linear_regression(
            clean_data,
            job["x_col"],
            job["y_col"],
            job["figure_name"],
        )
        regression_runs.append((job, clean_data, regression_result))

    stats_df = pd.DataFrame([run[2] for run in regression_runs])
    stats_df.to_csv(STATS_OUTPUT_PATH, index=False)
    print_regression_summary(stats_df)
    print(f"\nSaved regression stats to: {STATS_OUTPUT_PATH}")

    for index, (job, clean_data, regression_result) in enumerate(regression_runs):
        draw_relationship_plot(df, job, regression_result, clean_data)
        plt.show(block=index == len(regression_runs) - 1)


if __name__ == "__main__":
    main()
