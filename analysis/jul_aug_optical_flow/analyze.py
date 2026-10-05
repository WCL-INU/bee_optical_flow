"""Summarize all saved July/August optical-flow results without re-running flow."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "bee_count_output/yaml_jul_aug_all"
OUT = Path(__file__).resolve().parent
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
for folder in (TABLES, FIGURES):
    folder.mkdir(parents=True, exist_ok=True)

def save(df, name):
    df.to_csv(TABLES / f"{name}.csv", index=False, float_format="%.12g")

def corr(a, b, method="pearson"):
    pair = pd.concat([pd.Series(a), pd.Series(b)], axis=1).dropna()
    if method == "spearman":
        pair = pair.rank(method="average")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))

def lag1(values):
    a, b = values.iloc[:-1], values.iloc[1:]
    if a.std(ddof=0) == 0 or b.std(ddof=0) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def grouped(d, keys):
    g = d.groupby(keys, observed=True)
    a = g.agg(n=("video", "size"), days=("date", "nunique"),
              duration_sec=("duration_sec", "sum"),
              raw_flux=("total_raw_traffic_flux", "sum"),
              filtered_flux=("total_filtered_traffic_flux", "sum"),
              in_flux=("total_filtered_in_flux", "sum"),
              out_flux=("total_filtered_out_flux", "sum"),
              mean_rate=("rate", "mean"), median_rate=("rate", "median"),
              p90_rate=("rate", lambda x: x.quantile(.9)),
              zero_fraction=("rate", lambda x: (x == 0).mean()),
              mean_cv=("window_cv", "mean"),
              median_top4_share=("window_top4_share", "median"),
              mean_abs_balance=("abs_balance", "mean"))
    a["retention"] = a.filtered_flux / a.raw_flux.replace(0, np.nan)
    a["in_share"] = a.in_flux / a.filtered_flux.replace(0, np.nan)
    a["balance"] = 2 * a.in_share - 1
    return a.reset_index()

d = pd.read_csv(SOURCE / "batch_summary.csv")
assert not d.video.duplicated().any()
x = d.video.str.extract(r"ANU-25-summer-(\d+)_(\d{8})_(\d{6})\.mp4")
assert not x.isna().any().any()
d["device"] = x[0].astype(int)
d["date"] = pd.to_datetime(x[1])
d["hour"] = x[2].str[:2].astype(int)
d["month"] = d.date.dt.month
d["rate"] = d.total_filtered_traffic_flux / d.duration_sec
d["retention"] = d.total_filtered_traffic_flux / d.total_raw_traffic_flux.replace(0, np.nan)
d["in_share"] = d.total_filtered_in_flux / d.total_filtered_traffic_flux.replace(0, np.nan)
d["balance"] = 2 * d.in_share - 1
d["abs_balance"] = d.balance.abs()
window_cols = ["window", "start_sec", "end_sec", "raw_in_flux_sum", "raw_out_flux_sum",
               "filtered_in_flux_sum", "filtered_out_flux_sum", "filtered_traffic_count_est"]

def read_window(row):
    w = pd.read_csv(SOURCE / (Path(row.video).stem + "_window_3sec.csv"), usecols=window_cols)
    z = w.filtered_in_flux_sum + w.filtered_out_flux_sum
    raw = w.raw_in_flux_sum + w.raw_out_flux_sum
    ws = z.sum()
    info = {"video": row.video, "window_n": len(w), "window_mean": z.mean(),
            "window_std": z.std(ddof=0), "window_max": z.max(),
            "window_cv": z.std(ddof=0) / z.mean() if ws > 0 else np.nan,
            "window_zero_fraction": (z == 0).mean(),
            "window_top4_share": z.nlargest(4).sum() / ws if ws > 0 else np.nan,
            "window_max_mean": z.max() / z.mean() if ws > 0 else np.nan,
            "window_lag1": lag1(z),
            "window_balance_sd": ((w.filtered_in_flux_sum - w.filtered_out_flux_sum) /
                                  z.replace(0, np.nan)).std(ddof=0),
            "window_sum_abs_error": abs(ws - row.total_filtered_traffic_flux),
            "window_raw_sum_abs_error": abs(raw.sum() - row.total_raw_traffic_flux),
            "count_scale_abs_error": float(np.max(np.abs(z - w.filtered_traffic_count_est * 100))),
            "in_sum_abs_error": abs(w.filtered_in_flux_sum.sum() - row.total_filtered_in_flux),
            "out_sum_abs_error": abs(w.filtered_out_flux_sum.sum() - row.total_filtered_out_flux)}
    w["flux"] = z
    w["raw_flux"] = raw
    w["video"] = row.video
    w["device"] = row.device
    w["date"] = row.date
    w["hour"] = row.hour
    w["balance"] = (w.filtered_in_flux_sum - w.filtered_out_flux_sum) / z.replace(0, np.nan)
    return info, w

print(f"Reading all {len(d):,} window CSVs", flush=True)
infos, windows = [], []
with ThreadPoolExecutor(max_workers=4) as pool:
    for i, (info, w) in enumerate(pool.map(read_window, d.itertuples(index=False)), 1):
        infos.append(info)
        windows.append(w)
        if i % 3000 == 0:
            print(f"Windows {i:,}/{len(d):,}", flush=True)
d = d.merge(pd.DataFrame(infos), on="video", validate="one_to_one")
w = pd.concat(windows, ignore_index=True)
del windows
save(w.nlargest(30, "flux")[["video", "device", "date", "hour", "window", "start_sec", "end_sec",
                             "filtered_in_flux_sum", "filtered_out_flux_sum", "flux", "balance"]],
     "top_windows")
save(w.groupby("hour").agg(n=("flux", "size"), mean_flux=("flux", "mean"),
                          median_flux=("flux", "median"), p90_flux=("flux", lambda x: x.quantile(.9)),
                          zero_fraction=("flux", lambda x: (x == 0).mean())).reset_index(), "window_hour")

frame_cols = ["raw_in_flux", "raw_out_flux", "filtered_in_flux", "filtered_out_flux",
              "raw_candidate_pixels", "persistent_candidate_pixels", "filtered_candidate_pixels",
              "raw_component_count", "valid_component_count", "rejected_component_count"]

def read_frame(row):
    f = pd.read_csv(SOURCE / (Path(row.video).stem + "_frame_flux.csv"), usecols=frame_cols)
    result = {"video": row.video, "frame_n": len(f)}
    for col in frame_cols:
        result["frame_sum_" + col] = f[col].sum()
    z = f.filtered_in_flux + f.filtered_out_flux
    result["frame_active_fraction"] = (z > 0).mean()
    result["frame_bidirectional_fraction"] = ((f.filtered_in_flux > 0) & (f.filtered_out_flux > 0)).mean()
    result["frame_lag1"] = lag1(z)
    result["frame_peak_rate"] = z.max() * 24
    return result

if "--reuse-frames" in sys.argv:
    f = pd.read_csv(TABLES / "frame_video_summary.csv")
    assert set(f.video) == set(d.video)
    print("Reusing the frame aggregates from this analysis run", flush=True)
else:
    print(f"Reading all {len(d):,} frame CSVs", flush=True)
    frames = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, result in enumerate(pool.map(read_frame, d.itertuples(index=False)), 1):
            frames.append(result)
            if i % 2000 == 0:
                print(f"Frames {i:,}/{len(d):,}", flush=True)
    f = pd.DataFrame(frames)
d = d.merge(f, on="video", validate="one_to_one")
save(f, "frame_video_summary")
save(d[["video", "device", "date", "month", "hour", "duration_sec", "processed_frame_pairs",
        "total_raw_in_flux", "total_raw_out_flux", "total_filtered_in_flux", "total_filtered_out_flux",
        "total_raw_traffic_flux", "total_filtered_traffic_flux", "rate", "retention", "in_share", "balance",
        "abs_balance", "window_n", "window_cv", "window_zero_fraction", "window_top4_share",
        "window_max_mean", "window_lag1", "window_balance_sd", "frame_active_fraction",
        "frame_bidirectional_fraction", "frame_lag1"]], "video_summary")
device = grouped(d, "device")
hour = grouped(d, "hour")
month = grouped(d, "month")
dm = grouped(d, ["device", "month"])
day = grouped(d, "date")
dh = grouped(d, ["device", "hour"])
for name, data in [("device", device), ("hour", hour), ("month", month),
                   ("device_month", dm), ("day", day), ("device_hour", dh)]:
    save(data, name)

# Compare months within the same device/hour and weight each shared cell equally.
cell = d.groupby(["device", "hour", "month"]).rate.mean().unstack("month").dropna()
cell.columns = ["july", "august"]
cell["difference"] = cell.august - cell.july
cell["relative_change"] = cell.august / cell.july.replace(0, np.nan) - 1
save(cell.reset_index(), "matched_device_hour_month")
md = cell.groupby("device")[["july", "august"]].mean()
md["relative_change"] = md.august / md.july.replace(0, np.nan) - 1
md["hours"] = cell.groupby("device").size()
save(md.reset_index(), "matched_device_month")

# Complete days supply an equal 17-hour within-device reference for date comparisons.
dc = d.groupby(["device", "date"]).agg(hours=("hour", "nunique"), rate=("rate", "mean"),
                                             flux=("total_filtered_traffic_flux", "sum"))
dc = dc[dc.hours == 17].copy()
dc["device_reference"] = dc.groupby("device").rate.transform("median")
dc["relative_activity"] = dc.rate / dc.device_reference.replace(0, np.nan)
save(dc.reset_index(), "complete_device_days")
normalized_days = dc.groupby("date").agg(devices=("rate", "size"),
                                          median_relative=("relative_activity", "median"),
                                          mean_relative=("relative_activity", "mean"))
save(normalized_days.reset_index(), "normalized_day")
daily_pivot = dc.rate.unstack("device")
daily_cor = daily_pivot.corr(min_periods=10)
save(daily_cor.reset_index(), "daily_device_correlation")

d["activity_group"] = pd.qcut(d.rate.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
activity = grouped(d, "activity_group")
activity["positive_median_retention"] = d[d.total_raw_traffic_flux > 0].groupby("activity_group", observed=True).retention.median().reindex(activity.activity_group).to_numpy()
save(activity, "activity_quartiles")
positive = d[d.rate > 0]
zero = d[d.rate == 0]
total = float(d.total_filtered_traffic_flux.sum())
raw = float(d.total_raw_traffic_flux.sum())
quantiles = [.0, .1, .25, .5, .75, .9, .95, .99, 1]
distribution = pd.DataFrame({"quantile": quantiles})
for col in ["rate", "retention", "abs_balance", "window_cv", "window_top4_share", "window_max_mean",
            "window_lag1", "frame_active_fraction", "frame_lag1"]:
    distribution[col] = d[col].quantile(quantiles).to_numpy()
save(distribution, "distribution")
save(d.nlargest(20, "rate")[["video", "rate", "retention", "in_share", "window_cv", "window_top4_share"]], "top_videos")
save(positive.nlargest(15, "window_top4_share")[["video", "rate", "window_cv", "window_top4_share", "window_max_mean"]], "burst_videos")
check = {}
for col in ["raw_in", "raw_out", "filtered_in", "filtered_out"]:
    diff = (d["frame_sum_" + col + "_flux"] - d["total_" + col + "_flux"]).abs()
    check["max_frame_summary_error_" + col] = float(diff.max())
for col in ["window_sum_abs_error", "window_raw_sum_abs_error", "count_scale_abs_error", "in_sum_abs_error", "out_sum_abs_error"]:
    check["max_" + col] = float(d[col].max())
assert (d.frame_n == d.processed_frame_pairs).all()
assert (d.window_n == 40).all()
assert max(check.values()) < 0.01
check["filtered_exceeds_raw_rows"] = int((d.total_filtered_traffic_flux > d.total_raw_traffic_flux + 1e-6).sum())
check["window_count"] = len(w)
check["frame_count"] = int(d.frame_n.sum())

comp = pd.DataFrame({"device": d.device, "in_retention": d.total_filtered_in_flux / d.total_raw_in_flux.replace(0, np.nan),
                     "out_retention": d.total_filtered_out_flux / d.total_raw_out_flux.replace(0, np.nan)})
save(comp.groupby("device").median().reset_index(), "direction_retention")
pairs = daily_cor.to_numpy()[np.triu_indices(len(daily_cor), 1)]
pairs = pairs[np.isfinite(pairs)]
summary = {
    "n": len(d), "devices": int(d.device.nunique()), "days": int(d.date.nunique()),
    "start": str(d.date.min().date()), "end": str(d.date.max().date()),
    "observed_hours": float(d.duration_sec.sum() / 3600),
    "total_raw_flux": raw, "total_filtered_flux": total,
    "total_filtered_in": float(d.total_filtered_in_flux.sum()), "total_filtered_out": float(d.total_filtered_out_flux.sum()),
    "retention": total / raw, "removal": 1 - total / raw,
    "in_retention": float(d.total_filtered_in_flux.sum() / d.total_raw_in_flux.sum()),
    "out_retention": float(d.total_filtered_out_flux.sum() / d.total_raw_out_flux.sum()),
    "in_share": float(d.total_filtered_in_flux.sum() / total),
    "zero_n": len(zero), "zero_raw_zero_n": int((zero.total_raw_traffic_flux == 0).sum()),
    "zero_raw_positive_n": int((zero.total_raw_traffic_flux > 0).sum()),
    "window_zero_n": int((w.flux == 0).sum()),
    "window_both_positive_n": int(((w.filtered_in_flux_sum > 0) & (w.filtered_out_flux_sum > 0)).sum()),
    "positive_balance_under_10pct": float((positive.abs_balance < .1).mean()),
    "positive_in_dominant": int((positive.balance > 0).sum()),
    "positive_out_dominant": int((positive.balance < 0).sum()),
    "pearson_in_out": corr(d.total_filtered_in_flux, d.total_filtered_out_flux),
    "spearman_in_out": corr(d.total_filtered_in_flux, d.total_filtered_out_flux, "spearman"),
    "spearman_rate_retention_positive": corr(positive.rate, positive.retention, "spearman"),
    "spearman_rate_cv_positive": corr(positive.rate, positive.window_cv, "spearman"),
    "spearman_rate_abs_balance_positive": corr(positive.rate, positive.abs_balance, "spearman"),
    "burst_top4_over_half_n": int((positive.window_top4_share > .5).sum()),
    "burst_max_mean_over5_n": int((positive.window_max_mean > 5).sum()),
    "top_1pct_share": float(d.nlargest(int(np.ceil(len(d) * .01)), "rate").total_filtered_traffic_flux.sum() / total),
    "top_10pct_share": float(d.nlargest(int(np.ceil(len(d) * .1)), "rate").total_filtered_traffic_flux.sum() / total),
    "matched_cells": len(cell), "matched_devices": int(cell.index.get_level_values(0).nunique()),
    "matched_july_mean": float(cell.july.mean()), "matched_august_mean": float(cell.august.mean()),
    "matched_change": float(cell.august.mean() / cell.july.mean() - 1),
    "matched_devices_increased": int((md.relative_change > 0).sum()),
    "complete_device_days": len(dc),
    "daily_device_corr_median": float(np.median(pairs)),
    "daily_device_corr_pairs": len(pairs),
    "frame_active_fraction": float(d.frame_active_fraction.mean()),
    "frame_bidirectional_fraction": float(d.frame_bidirectional_fraction.mean()),
    "candidate_pixel_retention": float(d.frame_sum_filtered_candidate_pixels.sum() / d.frame_sum_raw_candidate_pixels.sum()),
    "persistence_pixel_retention": float(d.frame_sum_persistent_candidate_pixels.sum() / d.frame_sum_raw_candidate_pixels.sum()),
    "area_pixel_retention": float(d.frame_sum_filtered_candidate_pixels.sum() / d.frame_sum_persistent_candidate_pixels.sum()),
    "component_retention": float(d.frame_sum_valid_component_count.sum() / d.frame_sum_raw_component_count.sum()),
    "raw_mean_normal_flow_per_pixel": float(raw / d.frame_sum_raw_candidate_pixels.sum()),
    "filtered_mean_normal_flow_per_pixel": float(total / d.frame_sum_filtered_candidate_pixels.sum()),
    "check": check,
}
for metric in ["frame_sum_raw_candidate_pixels", "frame_sum_persistent_candidate_pixels", "frame_sum_filtered_candidate_pixels",
               "frame_sum_raw_component_count", "frame_sum_valid_component_count", "frame_sum_rejected_component_count"]:
    summary[metric] = int(d[metric].sum())
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 180
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
COLORS = ["#247BA0", "#E59E36"]

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(hour.hour, hour.mean_rate / 1000, "o-", color=COLORS[0], label="평균")
ax[0].plot(hour.hour, hour.median_rate / 1000, "o--", color=COLORS[1], label="중앙값")
ax[0].set(xlabel="파일명 촬영 시각", ylabel="filtered flux / 초 (천 단위)", title="시간대별 활동량", xticks=range(5, 22, 2))
ax[0].legend()
ax[1].plot(hour.hour, hour.in_share * 100, "o-", color=COLORS[0])
ax[1].axhline(50, color="gray", linestyle="--")
ax[1].set(xlabel="파일명 촬영 시각", ylabel="IN flux 비중 (%)", title="시간대별 방향 균형", xticks=range(5, 22, 2))
fig.tight_layout(); fig.savefig(FIGURES / "01_hourly_profile.png"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
s = device.sort_values("mean_rate")
ax[0].barh(s.device.astype(str), s.mean_rate / 1000, color=COLORS[0])
ax[0].set(xlabel="평균 filtered flux / 초 (천 단위)", ylabel="기기 번호", title="기기별 관측 활동량")
p = dh.pivot(index="device", columns="hour", values="mean_rate") / 1000
im = ax[1].imshow(p, aspect="auto", cmap="YlOrRd", origin="upper")
ax[1].set(xticks=range(len(p.columns)), xticklabels=p.columns, yticks=range(len(p.index)), yticklabels=p.index,
          xlabel="촬영 시각", ylabel="기기 번호", title="기기별 하루 활동 곡선")
fig.colorbar(im, ax=ax[1], label="천 flux / 초")
fig.tight_layout(); fig.savefig(FIGURES / "02_device_profile.png"); plt.close(fig)

fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax[0].plot(day.date, day.mean_rate / 1000, color=COLORS[0], label="관측 영상 전체 평균")
ax[0].set(ylabel="천 flux / 초", title="날짜별 활동량과 기기 내 상대 활동량")
ax[0].legend()
ax[1].plot(normalized_days.index, normalized_days.median_relative, color=COLORS[1], label="17개 시각이 모두 있는 기기·일의 중앙값")
ax[1].axhline(1, color="gray", linestyle="--")
ax[1].set(ylabel="기기별 일평균 중앙값 대비 배율", xlabel="촬영 날짜")
ax[1].legend(fontsize=9)
fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(FIGURES / "03_daily_activity.png"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(positive.rate / 1000, positive.retention * 100, s=5, alpha=.15, color=COLORS[0], rasterized=True)
ax[0].set(xscale="log", xlabel="filtered flux / 초 (천 단위, 로그 축)", ylabel="filtered / raw (%)", title="활동량과 필터 통과율")
ax[1].scatter(positive.rate / 1000, positive.window_cv, s=5, alpha=.15, color=COLORS[1], rasterized=True)
ax[1].set(xscale="log", xlabel="filtered flux / 초 (천 단위, 로그 축)", ylabel="3초 구간 변동계수", title="활동량과 구간 변동성")
fig.tight_layout(); fig.savefig(FIGURES / "04_retention_and_variability.png"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].hist(positive.balance, bins=70, color=COLORS[0])
ax[0].set(xlabel="(IN − OUT) / (IN + OUT)", ylabel="영상 수", title="양의 flux 영상의 방향 균형")
ax[0].axvline(0, color="gray", linestyle="--")
ax[1].bar(md.index.astype(str), md.relative_change * 100, color=np.where(md.relative_change > 0, COLORS[0], COLORS[1]))
ax[1].axhline(0, color="gray", linewidth=.8)
ax[1].set(xlabel="기기 번호", ylabel="8월 / 7월 변화율 (%)", title="기기별 공통 시각 평균의 월별 변화")
fig.tight_layout(); fig.savefig(FIGURES / "05_balance_and_month_change.png"); plt.close(fig)

examples = [d.loc[d.rate.idxmax(), "video"], positive.loc[positive.window_top4_share.idxmax(), "video"],
            positive.iloc[(positive.rate - positive.rate.median()).abs().argmin()].video]
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
for ax, video in zip(axes, examples):
    q = w[w.video == video]
    ax.plot(q.start_sec, q.filtered_in_flux_sum / 1000, label="IN", color=COLORS[0])
    ax.plot(q.start_sec, q.filtered_out_flux_sum / 1000, label="OUT", color=COLORS[1])
    ax.set(title=video, ylabel="3초 flux (천 단위)")
    ax.legend(loc="upper right")
axes[-1].set_xlabel("영상 내 경과 초")
fig.tight_layout(); fig.savefig(FIGURES / "06_example_windows.png"); plt.close(fig)
print("Analysis tables and six figures written", flush=True)
