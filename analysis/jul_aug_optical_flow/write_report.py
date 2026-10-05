"""Write Korean Markdown and a standalone HTML report from the analysis tables."""
from pathlib import Path
import base64
import html
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from report_assets import prepare_report_assets

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
T = HERE / "tables"
F = HERE / "figures"
REPORT_NAME = "optical_flow_jul_aug_analysis_report"
s = json.loads((HERE / "summary.json").read_text())
geo = json.loads((HERE / "geometry_summary.json").read_text())
gd = pd.read_csv(T / "geometry_device_comparison.csv")
gp = pd.read_csv(T / "geometry_periods.csv")
gc = pd.read_csv(T / "geometry_changes.csv")
gm = pd.read_csv(T / "geometry_month_groups.csv")
meaning = json.loads((HERE / "normalization_review.json").read_text())
meaning_cases = pd.read_csv(T / "normalization_review_cases.csv")
coverage = json.loads((HERE / "batch_summary_coverage.json").read_text())
d = pd.read_csv(T / "video_summary.csv", parse_dates=["date"])
device = pd.read_csv(T / "device.csv")
hour = pd.read_csv(T / "hour.csv")
month = pd.read_csv(T / "month.csv")
dm = pd.read_csv(T / "device_month.csv")
matched = pd.read_csv(T / "matched_device_month.csv")
cells = pd.read_csv(T / "matched_device_hour_month.csv")
dh = pd.read_csv(T / "device_hour.csv")
daily = pd.read_csv(T / "normalized_day.csv")
dist = pd.read_csv(T / "distribution.csv").set_index("quantile")
quart = pd.read_csv(T / "activity_quartiles.csv")
top = pd.read_csv(T / "top_videos.csv")
topw = pd.read_csv(T / "top_windows.csv")
positive = d[d.rate > 0]
all_flux = d.total_filtered_traffic_flux.sum()

def number(x, digits=0):
    return f"{x:,.{digits}f}"

def percent(x, digits=1):
    return f"{100*x:.{digits}f}%"

def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])

def link(name):
    return f"analysis/jul_aug_optical_flow/{name}"

def figure(name, caption):
    return f"![{caption}]({link('figures/'+name)})\n\n{caption}"

# Equal device weighting makes each device's hourly shape visible on the same scale.
profile = dh.pivot(index="device", columns="hour", values="mean_rate")
normalized_hour = profile.div(profile.mean(axis=1), axis=0).mean()
hour_extra = hour[["hour"]].copy()
hour_extra["equal_device_relative_rate"] = hour_extra.hour.map(normalized_hour)
shared_hour = cells.groupby("hour")[["july", "august"]].mean()
hour_extra["matched_july_rate"] = hour_extra.hour.map(shared_hour.july)
hour_extra["matched_august_rate"] = hour_extra.hour.map(shared_hour.august)
hour_extra.to_csv(T / "normalized_hour.csv", index=False)
device["peak_hour"] = device.device.map(profile.idxmax(axis=1))
device["mean_median_ratio"] = device.mean_rate / device.median_rate.replace(0, np.nan)
device.to_csv(T / "device_with_peak.csv", index=False)

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(normalized_hour.index, normalized_hour, "o-", color="#247BA0")
axes[0].axhline(1, color="gray", linestyle="--")
axes[0].set(xlabel="파일명 촬영 시각", ylabel="기기별 17개 시각 평균 대비 배율", title="각 기기를 같은 비중으로 본 활동 곡선")
axes[1].plot(shared_hour.index, shared_hour.july/1000, "o-", color="#247BA0", label="7월")
axes[1].plot(shared_hour.index, shared_hour.august/1000, "o-", color="#E59E36", label="8월")
axes[1].set(xlabel="파일명 촬영 시각", ylabel="천 flux / 초", title="공통 기기·시각의 월별 활동 곡선")
axes[1].legend()
for ax in axes:
    ax.set_xticks(range(5, 22, 2))
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(F / "07_normalized_hourly_comparison.png", dpi=180)
plt.close(fig)

peak = hour.loc[hour.mean_rate.idxmax()]
am_peak = hour[hour.hour.between(6, 12)].sort_values("mean_rate").iloc[-1]
mid = hour[hour.hour==14].iloc[0]
latest = pd.read_csv(ROOT / "validation/output/regression_model_comparison.csv")
legacy = pd.read_csv(ROOT / "validation/legacy/linear_regression_stats.csv")
old_rows = []
for label, records in [("기존 서술 보고서", [
    ["IN", 500, 0.9178, 0.8423, 12.00, 20.18, "1.16523e-5 × flux + 6.1251"],
    ["OUT", 500, 0.9033, 0.8159, 11.68, 18.67, "1.62326e-5 × flux + 4.9832"]])]:
    for r in records:
        old_rows.append([label, r[0], number(r[1]), f"{r[2]:.4f}", f"{r[3]:.4f}", number(r[4],2), number(r[5],2)])
for label, data in [("legacy 통계 CSV", legacy), ("현재 validation/output 선형 모델", latest[latest.model=="linear"])]:
    for r in data.itertuples(index=False):
        old_rows.append([label, r.y_col.upper(), number(r.n), f"{r.r:.4f}", f"{r.r_squared:.4f}", number(r.mae,2), number(r.rmse,2)])

hour_rows = [[f"{r.hour:02d}시", number(r.n), number(r.mean_rate/1000,1), number(r.median_rate/1000,1),
              percent(r.zero_fraction), percent(r.retention), percent(r.in_share,2)] for r in hour.itertuples(index=False)]
device_rows = [[r.device, number(r.n), number(r.days), number(r.mean_rate/1000,1), number(r.median_rate/1000,1),
                f"{r.peak_hour:02d}시", percent(r.zero_fraction), percent(r.retention)] for r in device.itertuples(index=False)]
matched_rows = [[r.device, number(r.july/1000,1), number(r.august/1000,1), f"{r.relative_change*100:+.1f}%"] for r in matched.itertuples(index=False)]
month_rows = [[f"{r.month}월", number(r.n), number(r.mean_rate/1000,1), number(r.median_rate/1000,1),
               percent(r.zero_fraction), percent(r.retention), percent(r.in_share,2)] for r in month.itertuples(index=False)]
q_rows = [[r.activity_group, number(r.n), number(r.median_rate/1000,1), percent(r.retention),
           number(r.mean_cv,3), percent(r.median_top4_share)] for r in quart.itertuples(index=False)]
top_rows = [[r.video.removesuffix(".mp4"), number(r.rate/1000,1), percent(r.retention),
             percent(r.in_share,2), number(r.window_cv,3), percent(r.window_top4_share)] for r in top.head(8).itertuples(index=False)]
date_high = daily[daily.devices>=10].nlargest(4, "median_relative")
date_low = daily[daily.devices>=10].nsmallest(4, "median_relative")
day_rows = [[r.date, number(r.devices), number(r.median_relative,3), "상위"] for r in date_high.itertuples(index=False)]
day_rows += [[r.date, number(r.devices), number(r.median_relative,3), "하위"] for r in date_low.itertuples(index=False)]
july = month[month.month==7].iloc[0]
august = month[month.month==8].iloc[0]
night21 = d[(d.hour==21)&(d.rate>0)]
signal_hours_share = d[d.hour.between(7,19)].total_filtered_traffic_flux.sum()/all_flux
late_share = d[d.hour.between(16,18)].total_filtered_traffic_flux.sum()/all_flux
quantile_rows = [[label, number(dist.loc[q,"rate"],1)] for label,q in [("P25",.25),("P50 / 중앙값",.5),("P75",.75),("P90",.9),("P95",.95),("P99",.99),("최댓값",1)]]
candidate_rows = [
    ["raw 후보 픽셀", number(s['frame_sum_raw_candidate_pixels']), "100.0%", "100.0%"],
    ["persistence 후 후보 픽셀", number(s['frame_sum_persistent_candidate_pixels']), percent(s['persistence_pixel_retention']), percent(s['persistence_pixel_retention'])],
    ["면적 필터 후 후보 픽셀", number(s['frame_sum_filtered_candidate_pixels']), percent(s['area_pixel_retention']), percent(s['candidate_pixel_retention'])]]

geometry_device_rows=[]
for r in gd.itertuples(index=False):
    p=gp[gp.device==r.device]
    dimensions=" / ".join(f"{int(q.ent_width)}×{int(q.ent_height)}" for q in p.itertuples(index=False))
    band=number(r.band_min) if r.band_min==r.band_max else f"{number(r.band_min)}–{number(r.band_max)}"
    geometry_device_rows.append([r.device,dimensions,band,number(r.mean_density,3),number(r.median_density,3),r.raw_rank,r.density_rank])
geometry_change_rows=[]
for r in gc.itertuples(index=False):
    geometry_change_rows.append([r.device,f"{number(r.old_band)} → {number(r.new_band)}",
        f"{100*r.band_change:+.2f}%",number(r.center_shift,1),percent(r.band_iou,2),
        f"{100*r.month_raw_change:+.1f}%",f"{100*r.month_density_change:+.1f}%"])
group_names={"all_common_devices":"공통 19개 기기", "unchanged_devices":"좌표가 유지된 기기", "changed_devices":"좌표가 변경된 기기", "exact_same_geometry":"정확히 같은 좌표끼리 비교"}
geometry_group_rows=[[group_names[r.group],r.devices,r.cells,number(r.rate_july/1000,1),number(r.rate_august/1000,1),
    f"{100*r.rate_change:+.1f}%",f"{100*r.density_change:+.1f}%"] for r in gm.itertuples(index=False)]
geometry_local_rows=[]
for r in gc.itertuples(index=False):
    a="해당 주 표본 없음" if r.local_common_hours==0 else f"{100*r.local_raw_change:+.1f}%"
    b="—" if r.local_common_hours==0 else f"{100*r.local_density_change:+.1f}%"
    geometry_local_rows.append([r.device,r.last_old_recording,r.first_new_recording,int(r.before_n),int(r.after_n),a,b])
meaning_labels=['같은 패치 1개: 좁은 경계','같은 패치 1개: 넓은 경계','같은 패치 2개: 넓은 경계',
                '국소 flow를 포함하는 경계','같은 면적의 경계를 이동하여 국소 flow 제외']
meaning_rows=[[label,number(r.boundary_pixels),number(r.filtered_flux),number(r.density,6)]
              for label,r in zip(meaning_labels,meaning_cases.itertuples(index=False))]

geometry_device_section=f"""### 6.2 면적 정규화의 측정 의미 검토

**면적 정규화를 실제 활동량의 보정으로 채택하지 않는다.** 현재 구현은 경계에서 검출된 움직임을 합산하므로, 움직임이 없는 픽셀을 더 포함한다고 flux가 그 면적에 비례해 커지는 구조가 아니다. 보고서의 주 지표는 기존 filtered flux이며, `flux/면적`은 측정 대상이 다른 공간 평균의 참고 통계로만 취급한다.

영상 전체 filtered traffic flux를 F, 처리 시간을 T, 경계 띠 픽셀 수를 A라 두면, 코드의 출력은 `F = Σ프레임 Σ후보픽셀 abs(normal_flow)`이고 공간 평균은 `D = F / (T × A)`이다. **F에는 필터를 통과한 후보만 기여하지만 D의 분모 A에는 움직임이 없는 픽셀도 포함**된다. 따라서 F/T는 경계 전체의 누적 신호이고 D는 비활성 픽셀까지 포함한 전체 경계의 픽셀당 평균 신호이다.

기존 목적은 입구 전체의 출입 활동을 나타내는 누적 신호를 얻는 것이었다. 넓은 입구에서 더 많은 움직임이 발생하는 차이를 A로 나누면 그 총량 차이가 사라질 수 있다. 반대로 같은 움직임을 유지하면서 비활성 경계만 길게 지정해도 D는 작아진다. **D의 순위 변화는 활동량이 더 정확하게 비교되었다는 증거가 아니라, 측정 대상이 바뀐 결과**이다. 기존 validation 회귀도 F를 입력으로 사용하므로 D의 보정 타당성을 검증한 자료가 아니다.

### 6.3 같은 합산 규칙으로 확인한 개념 검증

실제 코드의 경계 마스크, 강도 threshold, persistence 및 component 면적 필터를 사용해 두 조건을 독립적으로 바꾸었다. 첫째는 동일한 이동 패치를 유지하고 비활성 경계를 늘리는 조건, 둘째는 넓은 경계에 같은 이동 패치를 하나 더 추가하는 조건이다. **입력은 인위적으로 지정한 flow 벡터이며 실제 벌 영상이나 Farneback 성능의 검증 자료가 아니다.** 목적은 면적에 나누는 연산이 원래 합계의 의미를 항상 보존하는지 확인하는 것이다.

{table(['합성 flow 조건','경계 픽셀 수 A','filtered flux 합계 F','D = F/(T×A)'],meaning_rows)}

처음 세 사례의 처리 길이는 모두 2.125초이며, 이동 패치는 같은 크기·속도·경로를 가진다. 경계를 넓혀도 동일한 패치 1개의 flux는 **7,248로 동일**하지만 D는 **{percent(meaning['unchanged_passage_density_ratio'],2)}**로 낮아진다. 넓은 경계에서 패치가 2개로 늘면 F는 **2배**가 되지만 D는 좁은 경계의 패치 1개 사례 대비 **{percent(meaning['doubled_passages_density_ratio'],2)}**로 거의 같다. 총 이동 신호의 차이가 공간 평균에서 사라지는 구체적인 예이다.

마지막 두 사례는 같은 0.8333초의 국소 flow 입력에 같은 크기의 경계를 적용한 결과이다. 경계 면적은 동일한 9,452픽셀인데 위치를 이동하자 F는 **9,728에서 0**으로 바뀐다. 즉 **면적 변화율이 0이어도 위치 선택만으로 큰 flux 변화가 가능**하다. 실제 기기에서도 작은 면적 변화만 보고 flux 변화의 기여분이 작다고 판단할 수 없다.

{figure('11_normalization_meaning_review.png','그림 11. 합성 flow를 이용한 측정 의미 검토. 동일한 이동 패치의 누적량과 전체 경계 픽셀의 평균은 서로 다른 질문에 답한다.')}

이 계산은 면적 정규화가 항상 의미를 보존한다는 가정의 반례이다. 실제 데이터에서 F/A와 count의 관계를 검증한 실험이나 새 calibration을 수행한 것은 아니므로, 기존 활동량 대신 D를 채택하지 않는다.

### 6.4 공간 설정과 공간 평균의 참고 통계

합산 영역은 ENT 사각형 내부 전체가 아니라 **네 변 주변의 counting boundary band**이다. 이 영역의 크기를 기술할 때는 실제 경계 띠의 픽셀 수 A를 계산한다. 정확한 A를 아는 것과 A를 나누는 활동량 보정이 타당한 것은 별개의 문제이다.

저장된 완료 기록의 ROI/ENT 좌표를 영상별로 읽고, `build_entrance_mask()`와 `build_counting_boundary_band()`를 그대로 호출하여 A를 계산했다. 경계의 겹치는 모서리는 코드의 최근접 경계 배정에 따라 한 번씩 집계했다. 완료 기록이 있는 12,731개는 저장된 설정을 사용했고, 좌표 기록이 없는 957개는 파일명 시각에 해당하는 [roi_regions.yaml](roi_regions.yaml) 기간 좌표로 복원했다. 저장된 좌표와 현재 YAML의 일치는 확인했다.

이 자료에는 **26개 ROI/ENT 조합**이 있다. A는 **{number(geo['boundary_min'])}–{number(geo['boundary_max'])}픽셀**, 최대/최소 비는 **{geo['area_ratio']:.2f}배**이다. 8·12·13·14·16·18번은 두 좌표 기간을 가지며, 나머지 14개 기기는 하나의 좌표 기간을 가진다.

참고 통계 D의 단위는 `flux / (초·경계 픽셀)`이다. 아래 평균은 영상마다 해당 기간의 A로 나눈 후 평균했다. ENT 크기가 두 개이면 시간 순서로 나열했다. 두 정렬은 각각 경계 전체 신호와 공간 평균을 정렬한 것이며, 기기별 실제 출입량 순위로 해석하지 않는다.

{table(['기기','ENT 너비×높이(px)','실제 경계 픽셀 수','D 평균(참고)','D 중앙값(참고)','F/T 정렬','D 정렬'],geometry_device_rows)}

D가 높은 기기는 7·9·20번이며, 14번은 F/T 정렬에서 3번째, D 정렬에서 14번째이다. 이는 **픽셀 평균이 높은 경계와 누적 신호가 큰 경계가 다르다는 결과**이다. 14번의 활동량이 과대평가되어 순위를 바로잡았다는 결론으로 사용하지 않는다.

D는 `프레임 쌍/초 × 후보 픽셀 점유율 × 후보 픽셀당 평균 법선 flow`로 분해된다. 움직임의 픽셀 점유 정도와 강도를 평균한 통계로 사용할 수 있으며, 물리적 입구 길이당 벌의 통과 수로 환산한 값은 아니다. 기기 간 물리적 규모와 신호-활동량의 관계를 추가로 검증하기 전에는 F/T와 D 어느 쪽도 실제 출입량의 공통 척도로 확정하지 않는다.

{figure('08_geometry_normalized_devices.png','그림 8. 경계 전체의 누적 신호와 경계 픽셀의 평균 신호를 각각 정렬한 결과. 보정된 활동량 순위를 제시한 그림이 아니다.')}

기기별 분석은 관측 flux와 공간 설정을 함께 기술하고, 동일한 좌표 기간 안의 시간 변화에 우선 근거한다. D는 보조 통계로 표시한다.
"""

geometry_month_section=f"""### 7.3 ENT 변경 시점과 flux 변화의 동반 여부

[ROI/ENT YAML](roi_regions.yaml)에서 **8·12·13·14·16·18번은 8월 9일까지의 좌표와 8월 11일부터의 좌표가 구분**되어 있다. 완료 기록에서도 해당 좌표 변경이 확인된다. 8월 10일은 이 여섯 기기의 두 YAML 기간 사이에 있다.

아래 면적 변화는 변경 전/후 마스크의 실제 A를 비교한 값이다. 이동 거리와 IoU는 **원본 영상의 픽셀 좌표계**에서 계산했다. 월별 변화율은 각 기기의 공통 17개 시각을 같은 비중으로 평균했다.

{table(['기기','경계 픽셀 수: 전 → 후','A 변화','ENT 중심 이동(px)','경계 띠 IoU','월별 총 flux/초 변화','월별 D 변화(참고)'],geometry_change_rows)}

실제 합산 영역의 크기는 8번 −1.26%, 12번 −2.39%, 13번 −4.14%, 14번 −2.14%, 16번 +7.75%, 18번 −2.12% 바뀌었다. ENT 내부 면적과는 다른 변화이다. 예를 들어 **12번의 ENT 내부 면적은 +3.78%이지만 경계 띠는 −2.39%**, **16번의 ENT 내부 면적은 +19.60%이지만 경계 띠는 +7.75%**이다. ENT 내부 면적으로 나누면 실제 flux 합산 영역 변화와 다른 보정이 된다.

**16번은 합산 영역이 늘면서 관측 flux는 줄었다.** 총 flux/초는 −21.7%, D는 −27.3%이다. 이 두 수치의 차이는 나눗셈으로 발생한 차이이며, 넓어진 경계가 실제 감소를 가렸다는 근거로 쓰지 않는다. 13번과 14번의 F/T 증가율과 D 증가율의 차이도 같은 이유로, 면적의 인과 효과를 제거한 결과로 해석하지 않는다.

ENT 중심은 **29.2–99.4픽셀 이동**했고, 원본 영상 좌표에서의 경계 띠 IoU는 **0–5.55%**이다. 면적 변화 외에 **어떤 영상 위치에서 신호를 합산했는지도 변경**되었다. 이 값은 서로 다른 촬영 영상에서 동일한 물리적 입구가 얼마나 겹치는지를 측정한 값이 아니라, 지정된 영상 좌표 마스크의 겹침이다.

{figure('09_boundary_coordinate_changes.png','그림 9. 여섯 기기의 변경 전/후 실제 경계 띠를 원본 영상 좌표에 표시한 결과. 파랑은 변경 전, 주황은 변경 후이다.')}

### 7.4 면적 비례 가정의 검토와 동일 좌표 비교

실제 활동량을 복원하려는 보정식 `F/T × A기준/A현재`를 쓰려면, 적어도 다음 가정이 필요하다.

- 추가·제외되는 경계 영역이 기존 영역과 같은 정도의 신호를 담아, 기대 flux가 면적에 비례한다.
- 영역 변경이 서로 다른 출입 경로, 법선 방향 또는 이동 신호의 핵심 위치를 선택하는 변화가 아니다.
- 영상의 픽셀 스케일과 필터가 통과시키는 신호의 관계도 같은 방식으로 유지된다.

코드에는 flux와 A의 비례를 강제하는 규칙이 없고, 위 가정은 현재 저장 자료에서 검증되지 않았다. 기존 회귀는 F의 관계를 검증했으며 이 면적 보정식을 검증하지 않았다. 따라서 **7월 면적으로 환산한 −13.0%를 보정된 월별 활동량 결론으로 채택하지 않는다.**

이미 계산한 −13.0%는 **면적 비례를 가정했을 때의 민감도 계산**으로 남긴다. 공통 323개 기기·시각에 이 식을 적용하면 7월 {number(geo['july_reference_area_july_rate']/1000,1)}천, 8월 {number(geo['july_reference_area_august_rate']/1000,1)}천이라는 가상 기준 면적의 값이 나온다. 원래 F/T 변화 −13.4%와 약 0.4%p 차이가 있다는 것은 이 가정식의 수치적 영향이며, 실제 영역 변경의 기여분을 측정한 결과가 아니다.

{table(['비교 대상','기기 수','공통 조합 수','7월: 천 flux/초','8월: 천 flux/초','관측 F/T 변화','D 변화(참고)'],geometry_group_rows)}

추가적인 면적 비례 가정 없이 비교할 수 있는 자료는 **좌표가 유지된 13개 기기**와 **정확히 같은 좌표 기간의 관측끼리 묶은 조합**이다. 전자에서는 같은 시각 평균 F/T가 **19.0% 감소**, 후자의 16개 기기·272개 조합에서는 **15.8% 감소**한다. 이것은 같은 설정 경계에서 출력된 flux가 월 사이에 낮아지는 패턴이 있다는 관측 사실이다. 영역이 바뀐 다른 기기의 반사실적 flux를 복원한 수치는 아니다.

전체 공통 기기의 D 평균 변화 **−20.9%**는 공간 평균이라는 다른 지표의 변화이고, 기기별 A를 나누면서 집계 비중도 달라진다. 좌표가 바뀐 여섯 기기의 F/T 평균 변화 +0.1%와 D 평균 변화 −10.7%도 서로 다른 통계이다. 이 차이를 통해 실제 출입 활동량의 감소가 정정되었다고 결론 내리지 않는다.

### 7.5 변경 시점 부근의 관측 비교

변경 직전 달력주 **8월 3–9일**과 직후 달력주 **8월 11–17일**에서 각 기기의 공통 시각 평균을 비교했다. 아래 기록 날짜는 결과 디렉토리에서 확인되는 구좌표의 마지막 날짜와 신좌표의 첫 날짜이다.

{table(['기기','구좌표 마지막 관측','신좌표 첫 관측','직전 주 영상 수','직후 주 영상 수','근접 기간 총 flux/초 변화','근접 기간 D 변화'],geometry_local_rows)}

양쪽 달력주가 모두 관측된 **12·14·18번**은 각각 총 flux/초가 **−12.5%, +3.9%, −30.6%** 변했다. D 변화 −10.3%, +6.2%, −29.1%는 참고 통계이다. 이 세 기기에서는 좌표 기간 변경과 flux 변화가 시간상 함께 관측된다. **면적 변화가 작다는 이유로 설정 변경의 영향도 작다고 결론 내리지 않는다.** 추가·제외된 영역이 어떤 움직임을 담는지가 알려져야 그 기여분을 계산할 수 있다.

같은 두 달력주에서 관측된 **좌표가 유지된 9개 기기**의 공통 시각 평균도 총 flux/초가 **14.1%**, D가 **18.0% 감소**했다. 따라서 이 시기에는 ENT 변경 여부와 별개로 발생한 시간적 변화도 함께 나타난다.

8·13·16번은 직전 달력주의 자료가 없어 8월 11일 부근의 즉각적인 변화량을 계산하지 않았다. 이 세 기기의 월별 비교는 각각 7월 22일·7월 7일·7월 28일까지의 구좌표 자료와 8월 11일부터의 신좌표 자료를 사용하므로, **자료 공백을 사이에 둔 기간 차이**이다.

{figure('10_density_by_geometry_period.png','그림 10. 좌표가 변경된 여섯 기기의 픽셀 평균 신호 D 일평균(참고 통계). 점선은 8월 11일이며, 서로 다른 좌표 기간 사이는 연결하지 않았다.')}

확인된 것은 **설정 영역과 flux 변화의 동반 관측**, **같은 좌표에서의 월별 flux 변화**이다. 영역 변경이 실제 신호에 미친 영향은 현재 합계 CSV로 분리할 수 없다. 이를 직접 측정하려면 동일 영상 구간에 두 설정을 적용해 결과를 비교하고, 같은 이동이 공통 물리 경계에서 어떻게 포착되는지를 확인해야 한다. 기존 관측 flux와 설정 변경 기록은 유지하고, 검증되지 않은 면적 보정은 조건부 계산으로만 남겼다.
"""

sections = []
sections.append(f"""# 2026년 7–8월 벌통 입구 Optical Flow 결과 분석 보고서

작성일: 2026-10-05  
분석 대상: `bee_count_output/yaml_jul_aug_all`  
분석 전제: optical-flow 및 필터 파라미터는 동일하며, ROI/ENT 좌표는 기기와 기간별 설정을 사용한다. 주 지표는 기존 filtered flux이다. 면적 정규화는 측정 의미를 검토한 뒤 공간 평균의 참고 통계로만 남겼다.

## 1. 주요 결과

이번 자료의 대표적인 패턴은 **오후 17시의 활동 정점, 높은 IN/OUT 균형, 고활동 영상의 지속적인 flux, 기기별로 다른 월별 변화**이다.

- 영상 **{number(s['n'])}개**, 기기 **{s['devices']}개**, **{s['days']}일**을 전수 분석했다. 프레임 쌍은 **{number(s['check']['frame_count'])}개**, 3초 구간은 **{number(s['check']['window_count'])}개**이다.
- filtered traffic flux의 총합은 **{s['total_filtered_flux']:.6e}**이며, raw flux의 **{percent(s['retention'],2)}**가 남았다. 영상별 평균 활동량은 **{number(d.rate.mean(),1)} flux/초**, 중앙값은 **{number(d.rate.median(),1)} flux/초**이다.
- 전체 시간대 평균은 **{int(peak.hour)}시 {number(peak.mean_rate,1)} flux/초**로 가장 높다. 오전의 국소 정점은 **{int(am_peak.hour)}시**이고, 13–14시의 완만한 하락 후 16–18시에 다시 커지는 형태다.
- 전체 IN 비중은 **{percent(s['in_share'],2)}**, OUT 비중은 **{percent(1-s['in_share'],2)}**이다. 영상별 IN/OUT flux의 Pearson 상관은 **{s['pearson_in_out']:.4f}**이다.
- 실제 경계 띠 면적은 기기·기간에 따라 **{number(geo['boundary_min'])}–{number(geo['boundary_max'])}픽셀**로 다르지만, **flux가 그 면적에 비례한다는 가정은 검증되지 않았다**. `flux/면적`의 정렬을 보정된 활동량 순위로 해석하지 않는다.
- 같은 기기·같은 시각의 관측 flux/초 평균은 8월에 **{percent(-s['matched_change'])} 감소**한다. 좌표가 유지된 13개 공통 기기에서도 **19.0% 감소**한다. 이는 관측된 flux의 변화이다.
- **8·12·13·14·16·18번**은 8월 11일부터 좌표가 바뀐다. 실제 합산 영역 크기와 위치의 변화 및 flux 변화를 7.3–7.5절에서 함께 분석했다.
- 양의 flux가 있는 영상에서 활동량과 3초 구간 변동계수의 Spearman 상관은 **{s['spearman_rate_cv_positive']:.3f}**이다. 활동량이 클수록 영상 안에서 flux가 지속되는 경향이 강하다.

## 2. 분석 자료와 지표

### 2.1 분석 범위

파일명의 날짜와 시각을 이용하여 **{s['start']}부터 {s['end']}까지, 05–21시**의 결과를 기기·날짜·시각별로 묶었다. 모든 영상은 프레임 쌍 2,879개, 처리 길이 119.9583초, 3초 구간 40개로 구성된다. 프레임 시각의 간격은 1/24초이다. 처리 영상의 길이를 합하면 **{number(s['observed_hours'],2)}시간**이다.

`batch_summary.csv`, 13,688개의 `*_window_3sec.csv`, 13,688개의 `*_frame_flux.csv`를 모두 읽었다. 영상 요약값을 3초 구간 합계 및 프레임 합계와 대조해 집계 일치를 확인했다. 아래 총합은 **저장된 약 2분 영상들의 관측 flux 합계**이며, 촬영 시각 사이를 적분한 하루 전체 활동량과는 집계 단위가 다르다.

### 2.2 지표 정의

| 지표 | 계산 | 의미 |
| --- | --- | --- |
| raw IN/OUT flux | 강도 기준을 통과한 후보 픽셀의 양/음의 경계 법선 flow 합 | persistence와 면적 필터 적용 전 방향별 신호 |
| filtered IN/OUT flux | 필터를 통과한 후보 픽셀의 방향별 flow 합 | 이번 보고서의 중심 지표 |
| traffic flux | IN + OUT | 양방향 경계 활동량 |
| 활동량, rate | 영상의 filtered traffic flux / 처리 초 | 시간당 비교의 기본이 되는 초당 누적 신호 |
| 필터 통과율 | filtered traffic / raw traffic | 후보 신호 중 최종 출력에 남은 비중 |
| IN 비중 | IN / (IN + OUT) | 전체 신호에서 IN 방향이 차지하는 비중 |
| 방향 균형 B | (IN − OUT) / (IN + OUT) | 양수는 IN 우세, 음수는 OUT 우세 |
| 3초 구간 변동계수 CV | 40개 구간의 모집단 표준편차 / 평균 | 영상 안에서의 상대적인 시간 변동성 |
| 상위 4구간 집중도 | 가장 큰 4개 구간 flux / 영상 전체 flux | 약 12초에 모인 신호의 비중 |

분모가 0인 비율과 변동계수는 정의되는 자료만으로 요약했다. 총합 비율은 합계끼리 나눈 flux 가중 비율이고, 영상별 비율의 중앙값은 각 영상을 같은 비중으로 취급한 통계이다. 상위 4구간은 크기순으로 선택하며, 서로 이어진 구간일 필요는 없다.

코드의 `count_est`는 **방향별 flux / 100**으로 저장된다. 이번 보고서에서는 이 값을 flux와 동일한 신호의 환산 지표로 해석하고, 활동량 수치는 원래 flux 단위로 제시한다.

## 3. 기존 시스템 자료와의 연결

### 3.1 연산 방식

기존 [README](README.md), [시스템 설명](docs/presentation.md), [persistence 문서](docs/bee_entrance_persistence_filter.md)와 핵심 구현 [bee_entrance_count.py](src/bee_entrance_count.py)에 따르면, 시스템은 ROI의 연속 영상에서 Farneback optical flow를 계산하고 입구 사각형의 네 경계 방향으로 flow를 투영한다.

각 후보 픽셀에서 `normal_flow = dx × normal_x + dy × normal_y`를 계산한다. 양의 성분을 IN, 음의 성분의 절댓값을 OUT으로 합산한다. Raw는 이미 boundary band와 flow 강도 조건을 통과한 신호이다. Persistence는 프레임 사이에 이어지는 후보를 남기며, 면적 필터는 그 뒤의 연결 성분을 정리한다. 최종 결과는 프레임별 신호와 3초 누적으로 저장된다.

저장된 완료 기록에 기재된 연산값은 boundary band 8픽셀, magnitude 기준 1.0, normal-flow 기준 0.5, blur 5, persistence decay 0.65·threshold 1.3, 최소 component 면적 200픽셀이다. Farneback은 levels 4, winsize 21, iterations 3을 사용한다. 이 공통 파라미터와 별도로, ROI/ENT는 기기 및 기간별 좌표를 사용한다.

이 과정에서 flux는 **경계 방향 움직임의 공간·시간 누적량**을 나타낸다. 이번 보고서의 방향성은 이 법선 투영의 IN/OUT 정의를 따른다.

### 3.2 기존 회귀 결과

프로젝트에는 초기 서술 보고서, legacy 통계 CSV, 현재 validation 출력에 서로 다른 표본 규모의 회귀 결과가 남아 있다. 각각의 자료에 기록된 수치를 정리하면 다음과 같다. 아래 MAE/RMSE의 단위는 기존 검증 자료의 count이다.

{table(['기존 자료','방향','관측 수','R','R²','MAE','RMSE'], old_rows)}

자료: [초기 선형 회귀 보고서](validation/legacy/linear_regression_report.md), [legacy 회귀 통계](validation/legacy/linear_regression_stats.csv), [현재 회귀 비교 결과](validation/output/regression_model_comparison.csv).

현재 validation 출력의 선형식은 IN에서 `8.666068712e-6 × filtered_in_flux + 6.960286266`, OUT에서 `8.384146128e-6 × filtered_out_flux + 7.171417991`이다. 같은 자료의 flat-exponential 모델은 IN R² 0.7727, OUT R² 0.7481로, 선형 모델과 매우 가까운 적합 결과를 보였다.

기존 자료는 filtered flux가 출입 활동량과 함께 증가한다는 해석을 뒷받침한다. 이번 7–8월 분석은 그 맥락을 이용해 flux의 상대적인 크기와 시간 구조를 해석하며, 별도의 count 회귀를 수행하지 않았다.

### 3.3 기존 feature 보고서의 해석

[2026-06-08 feature 분석 보고서](analysis/video_features/final/analysis_report.md)는 과거 검증 자료에서 frame difference, flow 크기, 방향 균형 등이 count 예측 오차와 연결됨을 보고했다. 낮은 flow에서의 과소예측 사례는 IN 279개, OUT 267개였고, `frame_diff_mean_p90`과 IN 과소예측 오차의 Pearson 상관은 0.758이었다. 방향 균형과 방향 분산도 관련 feature로 제시되었다.

이 기존 결과는 신호의 **총량뿐 아니라 시간 변화와 방향 구조도 함께 읽는 이유**를 제공한다. 실측 분석에서는 현재 디렉토리에 저장된 flux와 후보 픽셀·component 통계를 사용했다. 별도로 6.3절의 합성 flow는 면적 정규화의 의미를 검토하는 개념 계산이며, 실측 자료와 섞어 집계하지 않았다.
""")

sections.append(f"""## 4. 전체 활동량의 분포

전체 raw flux는 **{s['total_raw_flux']:.6e}**, filtered flux는 **{s['total_filtered_flux']:.6e}**이다. Filtered IN은 **{s['total_filtered_in']:.6e}**, OUT은 **{s['total_filtered_out']:.6e}**로 집계되었다.

{table(['영상별 활동량 통계','filtered traffic flux / 초'], quantile_rows)}

평균 {number(d.rate.mean(),1)}은 중앙값 {number(d.rate.median(),1)}보다 높다. 상위 10% 영상이 전체 filtered flux의 **{percent(s['top_10pct_share'])}**, 상위 1%가 **{percent(s['top_1pct_share'])}**를 차지한다. 이는 다수의 중간 활동 영상과 일부 큰 활동 영상이 함께 있는 오른쪽 꼬리 분포를 나타낸다. 상위 비중 계산에는 각각 활동량 상위 1,369개와 137개 영상을 사용했다.

Filtered flux가 0인 영상은 **{number(s['zero_n'])}개({percent(s['zero_n']/s['n'])})**이다. 이 가운데 raw도 0인 영상은 2,015개이고, raw는 있으나 최종 필터 후 0이 된 영상은 340개이다. 양의 filtered flux를 갖는 **{number(len(positive))}개** 영상의 활동량 중앙값은 **{number(positive.rate.median(),1)} flux/초**이다.

따라서 전체 중앙값, 양의 flux 영상의 중앙값, 큰 활동 구간의 상위 분위수는 서로 다른 활동 상태를 요약한다. 이후 해석에서는 0 구간이 많은 시간대와 지속적인 양의 flux 시간대를 함께 표시한다.

## 5. 시간대별 활동 곡선

{table(['촬영 시각','영상 수','평균: 천 flux/초','중앙값: 천 flux/초','0 영상 비중','통과율','IN 비중'], hour_rows)}

{figure('01_hourly_profile.png','그림 1. 전체 관측의 시간대별 활동량과 flux 가중 IN 비중.')}

시간대별 곡선은 6시부터 빠르게 상승해 오전 9시에 국소 정점을 만들고, 13–14시에 완만하게 낮아진 뒤 17시에 가장 높은 값을 보인다. **17시 평균은 9시의 {peak.mean_rate/am_peak.mean_rate:.2f}배, 14시의 {peak.mean_rate/mid.mean_rate:.2f}배**이다. 16–18시의 세 시각에 전체 관측 flux의 **{percent(late_share)}**가 모이며, 7–19시에는 **{percent(signal_hours_share)}**가 모인다.

5시, 20시, 21시는 중앙값이 모두 0이다. 특히 21시는 806개 중 800개가 0이고, 나머지 6개만 양의 flux를 갖는다. 이 6개 안의 높은 신호 때문에 21시 평균이 20시 평균보다 조금 높게 나타난다. 21시의 대표적인 상태는 중앙값 0과 99.3%의 0 비중으로 설명된다.

각 기기의 17개 시각 평균을 같은 비중으로 평균한 값을 기준으로 시간대 곡선을 나눈 뒤, 20개 기기를 같은 비중으로 평균해도 17시가 가장 높다. 이 값은 기기별 기준 활동량의 **{normalized_hour.loc[17]:.2f}배**이며, 20개 중 **{int((device.peak_hour==17).sum())}개**의 평균 곡선 정점이 17시에 있다. 정점이 16–18시에 있는 기기는 **{int(device.peak_hour.between(16,18).sum())}개**이다. 오후 정점은 여러 기기에서 반복되는 구조로 해석할 수 있다.

{figure('07_normalized_hourly_comparison.png','그림 2. 기기를 같은 비중으로 본 상대 시간대 곡선과 공통 기기·시각의 7월/8월 곡선.')}

가능한 해석은 **입구 경계 활동이 오전 상승과 오후 강화라는 일중 리듬을 갖는다**는 것이다. 낮 전체가 같은 수준으로 유지되기보다 시각에 따라 활동의 크기가 달라지며, 방향 비중은 대체로 50% 가까이 유지된다.

## 6. 기기별 활동 수준과 특징

### 6.1 관측 경계 전체의 flux

아래 활동량은 각 기기의 저장된 모든 시각을 포함한 **기존 관측 flux** 평균이다. 정점 시각은 기기별 시간대 평균이 가장 큰 시각으로 정의했다. 기기마다 합산 영역이 다르므로 이 정렬을 실제 출입량의 공통 순위로 확정하지 않는다. 6.2–6.3절에서 정규화의 의미를 먼저 검토하고, 6.4절에 다른 측정 대상인 픽셀 평균의 참고 통계를 제시한다.

{table(['기기','영상 수','관측 날짜 수','평균: 천 flux/초','중앙값: 천 flux/초','평균 곡선 정점','0 비중','통과율'], device_rows)}

{figure('02_device_profile.png','그림 3. 기기별 평균 활동량과 기기·시각별 활동량 지도.')}

관측 총 flux 평균이 높은 기기는 **9번 335.2천, 2번 298.1천, 14번 272.4천, 6번 235.3천 flux/초**이다. 9번과 2번은 중앙값도 각각 361.9천, 286.7천으로 높다. 이 수치는 해당 기기의 설정된 경계 전체에서 많은 관측에 걸쳐 큰 flux가 지속됨을 나타낸다.

1번과 19번의 평균은 각각 36.2천, 26.0천 flux/초이다. 19번 중앙값은 1.4천으로 평균과 차이가 크다. 17번도 평균 69.8천에 비해 중앙값은 10.9천이다. 이들 결과는 **대표적인 낮은 신호 상태와 일부 큰 신호 상태가 섞인 활동 분포**를 보여준다.

기기별 통과율은 19번 77.9%부터 7번 88.0%까지 분포한다. 높은 활동 수준의 기기와 낮은 활동 수준의 기기는 남는 신호의 비중도 서로 다르다. 기기의 특징을 표현할 때 평균 활동량, 중앙값, 0 비중, 통과율을 함께 읽으면 지속적인 활동과 간헐적인 활동을 구분하기 쉽다.

{geometry_device_section}

## 7. 7월과 8월의 변화

### 7.1 월별 관측 요약

{table(['월','영상 수','평균: 천 flux/초','중앙값: 천 flux/초','0 비중','통과율','IN 비중'], month_rows)}

전체 관측의 평균 활동량은 7월 168.8천에서 8월 134.9천 flux/초로 **{percent(1-august.mean_rate/july.mean_rate)} 감소**했다. 중앙값은 137.7천에서 77.7천으로 낮아졌고, 0 영상 비중은 15.8%에서 19.0%로 높아졌다. 전체 관측으로 보면 8월은 낮은 신호 상태의 비중이 더 크다.

### 7.2 같은 기기·시각의 비교

7월과 8월 모두 자료가 있는 **19개 기기 × 17개 시각 = 323개 조합**에서 각 월의 평균 활동량을 구했다. 각 조합에 동일한 가중치를 부여하면 7월은 **{number(s['matched_july_mean']/1000,1)}천**, 8월은 **{number(s['matched_august_mean']/1000,1)}천 flux/초**이고 변화율은 **{s['matched_change']*100:+.1f}%**이다. 기기별 표도 공통 17개 시각을 같은 비중으로 평균했다. 5번은 8월 자료만 있으므로 월별 짝 비교에는 들어가지 않는다.

{table(['기기','7월: 천 flux/초','8월: 천 flux/초','변화율'], matched_rows)}

증가 폭이 큰 기기는 **14번 +43.1%, 6번 +19.4%, 4번 +19.3%**이다. 13번과 2번도 증가했고, 7번은 +0.3%로 거의 같은 수준이다. 감소 폭이 큰 기기는 **17번 −76.5%, 10번 −67.8%, 11번 −63.5%, 19번 −61.5%**이다.

가능한 해석은 **8월의 전체적인 활동 감소와 기기별 증가가 동시에 존재한다**는 것이다. 월별 변화는 여러 입구에서 같은 크기로 나타나는 변화가 아니며, 기기별 경계 활동의 변화 폭과 방향이 다르다. 두 달의 공통 시각 곡선에서는 오후 정점이 계속 나타난다.

공통 기기·시각의 곡선을 시각별로 비교하면 8월의 **12시는 {percent(shared_hour.loc[12,'august']/shared_hour.loc[12,'july']-1)} 증가**, **13시는 {percent(shared_hour.loc[13,'august']/shared_hour.loc[13,'july']-1)} 증가**한다. 17시는 **{percent(1-shared_hour.loc[17,'august']/shared_hour.loc[17,'july'])} 감소**한다. 전체 평균이 낮아진 가운데 정오 부근의 신호는 커졌으므로, 월별 변화에는 활동 규모의 감소와 하루 안의 활동 분포 변화가 함께 들어 있다.

{geometry_month_section}
""")

sections.append(f"""## 8. 날짜별 활동 변화

각 기기에서 05–21시의 17개 시각이 모두 있는 기기·일 **{number(s['complete_device_days'])}개**를 골라 일평균 활동량을 계산했다. 이를 해당 기기의 완전한 기기·일 활동량 중앙값으로 나누어 상대 활동량을 구하고, 같은 날짜의 기기들에서 중앙값을 취했다. 값 1은 각 기기의 관측 기간 중 대표적인 일평균 활동 수준을 뜻한다.

본 절의 날짜 곡선은 관측 총 flux의 상대 변화이다. 그림 10의 D 곡선은 다른 지표인 공간 평균의 참고 통계이며, 이 곡선을 보정한 실제 활동량으로 취급하지 않는다.

아래 표는 이 집계에서 10개 이상의 기기가 참여한 날짜 가운데 상대 활동량 상위/하위 각 4일이다.

{table(['날짜','참여 기기 수','상대 활동량 중앙값','구분'], day_rows)}

{figure('03_daily_activity.png','그림 4. 날짜별 전체 관측 평균과 완전한 기기·일의 상대 활동량 중앙값.')}

7월 11일은 17개 기기의 상대 활동량 중앙값이 **1.435**로 대표 수준보다 약 43.5% 높고, 8월 16일은 13개 기기에서 **0.603**으로 약 39.7% 낮다. 여러 기기에서 활동 수준이 함께 높거나 낮은 날짜가 관측된다.

완전한 기기·일의 일평균 활동량을 공통 날짜 10일 이상인 기기 쌍에서 비교하면, 141개 쌍의 Pearson 상관 중앙값은 **{s['daily_device_corr_median']:.3f}**이다. 이 값과 기기별 월별 변화 결과를 함께 보면, 날짜에 따른 공통 활동 변화와 기기 고유의 변화가 함께 나타나는 것으로 해석할 수 있다. 여기서 날짜별 해석은 계산된 flux의 시간적 패턴에 근거한다.

## 9. IN/OUT 균형과 방향성

전체 flux에서 IN 비중은 **49.90%**, OUT 비중은 **50.10%**이다. 전체 방향 균형은 **{(2*s['in_share']-1):+.5f}**로 매우 작다. 영상별 IN과 OUT 총량은 Pearson **{s['pearson_in_out']:.4f}**, Spearman **{s['spearman_in_out']:.4f}**의 높은 상관을 보인다.

양의 flux 영상 가운데 **{percent(s['positive_balance_under_10pct'])}**는 `|B| < 0.1`, 즉 IN 비중이 45–55% 범위에 있다. `|B|` 중앙값은 **{dist.loc[.5,'abs_balance']:.4f}**이고 P90은 **{dist.loc[.9,'abs_balance']:.4f}**이다. IN 우세 영상은 **{number(s['positive_in_dominant'])}개**, OUT 우세 영상은 **{number(s['positive_out_dominant'])}개**이다. 방향 우세 영상의 개수와 전체 누적 flux는 각각 영상 단위와 신호 크기 단위의 요약이다.

3초 구간 전체에서 IN과 OUT이 동시에 양수인 구간은 **{number(s['window_both_positive_n'])}개({percent(s['window_both_positive_n']/s['check']['window_count'])})**이며, 양의 traffic flux 구간만 놓으면 **{percent(s['window_both_positive_n']/(s['check']['window_count']-s['window_zero_n']))}**이다. 프레임에서는 전체의 **{percent(s['frame_bidirectional_fraction'])}**가 양방향 양의 flux를 함께 가진다.

이 결과는 **높은 활동량에서 두 방향 신호가 함께 커지는 양방향 경계 활동**으로 해석할 수 있다. 기기 전체의 IN 비중도 6번 48.55%에서 18번 51.30% 범위로 좁다. 시간대별로 18–19시는 약한 IN 우세가 나타나지만, 큰 방향 이동보다는 traffic 크기의 변화가 자료의 주요 변동을 설명한다.

양의 flux 영상에서 활동량과 `|B|`의 Spearman 상관은 **{s['spearman_rate_abs_balance_positive']:.3f}**이다. 낮은 활동량에서는 한 방향 성분이 상대적으로 크게 보이는 구간이 더 많고, 높은 활동량에서는 방향 비율이 50% 부근에 모이는 경향이 있다.

{figure('05_balance_and_month_change.png','그림 5. 영상별 방향 균형의 분포와 기기별 공통 시각 평균의 월별 변화.')}

## 10. 필터링 결과의 수치적 특성

### 10.1 남는 flux와 후보 픽셀

전체 raw 대비 filtered flux 통과율은 **{percent(s['retention'],2)}**, 제거 비중은 **{percent(s['removal'],2)}**이다. IN 통과율은 **{percent(s['in_retention'],2)}**, OUT 통과율은 **{percent(s['out_retention'],2)}**로 비슷하다.

프레임 CSV의 후보 픽셀 수를 전부 더하면 다음과 같다. 아래 수는 같은 위치라도 프레임마다 다시 누적한 **픽셀·프레임 수**이다.

{table(['단계','누적 후보 픽셀·프레임','직전 단계 대비 통과율','raw 대비 통과율'], candidate_rows)}

Persistence 단계에서 raw 후보 픽셀의 약 **15.8%**가 제거되고, 면적 필터에서 그 뒤 남은 후보의 약 **9.8%**가 추가로 제거된다. 최종 후보 픽셀은 raw의 **76.0%**이다.

면적 필터 입력 component는 **{number(s['frame_sum_raw_component_count'])}개**, 통과 component는 **{number(s['frame_sum_valid_component_count'])}개**, 제거 component는 **{number(s['frame_sum_rejected_component_count'])}개**이다. Component 수의 통과율은 **{percent(s['component_retention'],2)}**이며, 이 component는 persistence 뒤 면적 필터에 입력된 연결 성분을 뜻한다.

결과적으로 **component 수는 61.7%, 후보 픽셀 수는 24.0%, flux는 15.3% 줄었다**. 남은 후보 픽셀의 평균 법선 flow 절댓값은 raw **{s['raw_mean_normal_flow_per_pixel']:.3f}**에서 filtered **{s['filtered_mean_normal_flow_per_pixel']:.3f}**로 커진다. 이는 제거된 성분이 평균적으로 더 작거나 약한 flow를 가진다는 해석과 맞는다. 통과한 component와 픽셀에는 더 큰 움직임 성분이 집중되어 있다.

### 10.2 활동 수준별 필터 통과율

영상별 활동량으로 전체 영상을 동일한 개수의 네 그룹으로 나누었다. Q1은 낮은 활동부터, Q4는 높은 활동까지이며 각 그룹은 3,422개이다. CV와 상위 4구간 집중도는 양의 flux 영상에서 정의된다.

{table(['활동 그룹','영상 수','중앙값: 천 flux/초','그룹 합계 통과율','CV 평균','상위 4구간 집중도 중앙값'], q_rows)}

Q1의 통과율은 **35.2%**, Q4는 **86.6%**이다. 양의 flux 영상의 활동량과 통과율 간 Spearman 상관은 **{s['spearman_rate_retention_positive']:.3f}**이다. 낮은 활동 그룹에서는 짧고 작은 후보가 차지하는 비중이 크고, 높은 활동 그룹에서는 필터 조건을 통과하는 지속적 신호가 더 큰 비중을 차지하는 것으로 해석할 수 있다.

{figure('04_retention_and_variability.png','그림 6. 양의 flux 영상의 활동량과 필터 통과율, 3초 구간 변동계수.')}

## 11. 3초 구간의 변동성과 큰 활동 사례

### 11.1 지속성과 간헐성

3초 구간에서 flux가 0인 구간은 **{number(s['window_zero_n'])}개({percent(s['window_zero_n']/s['check']['window_count'])})**이다. 양의 flux 영상의 CV 중앙값은 **{dist.loc[.5,'window_cv']:.3f}**, P90은 **{dist.loc[.9,'window_cv']:.3f}**이다. 상위 4구간 집중도 중앙값은 **{percent(dist.loc[.5,'window_top4_share'])}**로, 대표적인 영상에서는 약 10%의 시간에 16.2%의 flux가 모인다.

가장 큰 4개 구간에 전체 flux의 절반을 넘게 모은 영상은 **{number(s['burst_top4_over_half_n'])}개**, 양의 flux 영상의 **{percent(s['burst_top4_over_half_n']/len(positive))}**이다. 최대 구간이 구간 평균의 5배를 넘는 영상은 **{number(s['burst_max_mean_over5_n'])}개({percent(s['burst_max_mean_over5_n']/len(positive))})**이다. 이러한 영상은 한 영상 안에서도 신호가 간헐적으로 발생하는 유형이다.

활동량과 CV의 Spearman 상관 **−0.874** 및 활동 그룹별 결과는 고활동 영상의 신호가 상대적으로 균일하고 지속적임을 보여준다. Q4 CV 평균은 0.224, 상위 4구간 집중도 중앙값은 13.8%이며, Q1에서는 각각 2.319와 59.9%이다.

프레임별 양의 flux 비중은 전체에서 **{percent(s['frame_active_fraction'])}**이다. 연속 프레임 flux의 lag-1 상관은 영상별 중앙값 **{dist.loc[.5,'frame_lag1']:.3f}**, 연속 3초 구간에서는 **{dist.loc[.5,'window_lag1']:.3f}**이다. 프레임 단계의 persistence가 포함된 최종 출력은 가까운 시간끼리 이어지는 구조를 가지며, 3초 누적에서도 그 연속성이 일부 유지된다.

### 11.2 활동량 상위 영상

{table(['영상 식별자','천 flux/초','통과율','IN 비중','CV','상위 4구간 비중'], top_rows)}

가장 큰 영상은 **13번 기기의 8월 31일 17시**이며, 활동량은 **4,452.2천 flux/초**이다. 이 영상의 CV는 0.187, 상위 4구간 비중은 15.2%로 영상 전체에서 큰 신호가 이어진다. 다음으로 큰 **14번 기기의 7월 19일 07시**도 CV 0.142, 상위 4구간 비중 12.3%로 지속적인 고활동 사례이다.

전체 최대 3초 구간은 13번 기기의 위 영상 **114–117초**에서 나타났으며, filtered traffic flux는 **{number(topw.iloc[0].flux,1)}**, IN은 **{number(topw.iloc[0].filtered_in_flux_sum,1)}**, OUT은 **{number(topw.iloc[0].filtered_out_flux_sum,1)}**이다. 양방향 신호가 모두 크다.

큰 활동 영상에서도 개별 방향 편향은 나타난다. **17번 기기의 8월 16일 12시**는 IN 비중 **67.1%**, **2번 기기의 8월 3일 17시**는 OUT 비중 **58.6%**이다. 전체적으로 균형 잡힌 flux와 특정 영상에서의 방향성 있는 움직임이 함께 관측되는 사례이다.

상위 영상 대부분에서 통과율이 95% 이상이고 CV가 비교적 작다는 점은, 큰 flux가 작은 시간 구간의 단발성 성분만으로 구성되지 않고 많은 구간에서 유지된다는 해석을 뒷받침한다.

{figure('06_example_windows.png','그림 7. 최대 활동 영상, 상위 4구간 집중도가 100%인 간헐 영상, 양의 활동량 중앙값 근처 영상의 IN/OUT 시계열. 패널별 세로축 범위는 각 신호 크기를 따른다.')}

## 12. 종합 해석

이번 결과에서 optical flow가 보여주는 활동의 핵심은 다음 다섯 가지이다.

1. **하루 안의 반복적인 활동 리듬:** 오전 상승 뒤 오후 16–18시에 강해지며 17시가 대표 정점이다. 기기별 상대 곡선에서도 같은 구조가 나타난다.
2. **전체량이 함께 커지는 양방향 움직임:** IN과 OUT이 매우 높은 상관을 보이고, 대부분의 양의 flux 영상에서는 방향 비중이 50% 부근이다. 활동량 변화의 중심은 traffic 규모이다.
3. **지속적인 고활동과 간헐적인 저활동:** 활동량이 높은 영상은 낮은 상대 변동성과 높은 필터 통과율을 가진다. 낮은 활동에서는 작은 수의 3초 구간에 신호가 집중되는 경우가 많다.
4. **관측 flux의 월별 감소와 기기별 차이:** 공통 기기·시각의 평균 flux/초는 8월에 13.4% 감소했다. 같은 좌표에서의 비교에도 감소 패턴이 나타나며, 기기별 증가·감소는 서로 다르다. 면적 비례 가정의 −13.0%를 보정된 활동량 결론으로 채택하지 않는다.
5. **필터가 남긴 신호의 특성:** 많은 작은 component와 약한 후보 픽셀이 제거되면서도 전체 flux의 84.7%가 유지된다. 필터 후에는 후보 픽셀당 평균 경계 방향 움직임이 더 크다.

**면적을 나누면 누적량에서 공간 평균으로 측정 대상이 바뀐다.** 검증되지 않은 면적 비례 가정으로 기기별 활동량 순위나 월별 활동량을 정정하지 않는다. 기본 결과는 기존 flux의 규모·방향·지속성·시간 구조로 기술한다. 기기 간에는 공간 설정 차이를 함께 기록하며, 시간 변화는 동일 기기·동일 좌표 기간의 비교를 우선한다. 좌표가 바뀐 기간의 flux 차이는 설정과 시간 변화가 함께 들어 있는 관측으로 남긴다.

## 부록. 산출물과 재현

원본 배치 요약과 제외 목록의 공유용 사본은 [원본 요약 자료](bee_count_output/yaml_jul_aug_all)에 있다. 분석 자료와 보고서에서 연결하는 엑셀·기존 문서·그림·설정의 사본은 [analysis/jul_aug_optical_flow](analysis/jul_aug_optical_flow)에 모았다. 보고서의 모든 내부 파일 링크는 이 디렉토리 안의 공유 자료를 가리킨다.

### batch_summary 포함 여부와 엑셀 자료

프로젝트 최상단에 작성한 [optical_flow_jul_aug_analysis_data.xlsx](optical_flow_jul_aug_analysis_data.xlsx)의 공유용 사본을 분석 디렉토리에도 저장했다. 원본 `batch_summary.csv`의 **{number(coverage['batch_rows'])}행, {coverage['selected_column_count']}개 컬럼**을 추출했다. 영상 파일명·경로·처리 시간·프레임 쌍 수, ROI/ENT 좌표·경계 띠 폭, raw/filtered IN·OUT·traffic flux 합계와 초당 평균, raw/filtered 비율, 프레임·3초 구간 CSV 경로를 수록했다. 영상명 중복과 핵심 flow 컬럼의 결측은 모두 0이다. 엑셀을 다시 열어 {number(coverage['verified_export_cells'])}개 셀을 원본과 대조했다.

`videos` 디렉토리의 실제 7–8월 원본 파일명, 현재 ROI YAML의 기기·기간, `skipped_videos.csv`, 프레임·3초 구간 결과 파일 목록 및 배치 요약의 영상명을 대조한 결과는 다음과 같다.

{table(['월', '원본 영상', 'YAML 포함 영상', 'batch_summary', '프레임 CSV', '3초 구간 CSV', '제외 기록', '미확인 원본'], [[r['month']] + [number(r[k]) for k in ['source_videos', 'yaml_eligible', 'batch_videos', 'frame_results', 'window_results', 'skipped_videos', 'unaccounted_videos']] for r in coverage['monthly']])}

**optical flow를 추출한 {number(coverage['batch_rows'])}개 영상은 모두 `batch_summary.csv`에 존재한다.** 프레임 결과와 3초 구간 결과 각각의 영상명 집합도 배치 요약과 정확히 일치하며, 요약에 기록된 결과 CSV 경로가 모두 실제 파일과 연결된다. 원본 {number(coverage['source_videos'])}개 가운데 요약에 없는 {number(coverage['skipped_videos'])}개는 모두 **2026년 8월 10일의 8·12·13·14·16·18번 기기 영상**이다. 이날은 YAML 기간에 좌표 설정이 없으며, 80개 파일명이 기존 제외 기록과 정확히 일치한다. 따라서 처리 대상 중 요약 누락은 0개이고, 전체 원본 중 80개는 optical flow 미추출 영상이다.

엑셀의 `Optical flow` 시트는 원본 측정값을 그대로 담고, `컬럼 설명`, `포함 여부 확인`, `월별 포함 현황`, `추출 제외 영상` 시트에 정의와 대조 결과를 담았다. 원본의 ROI/ENT 설정 공란 {number(coverage['geometry_blank_rows'])}행은 그대로 보존했다. 면적 정규화 값은 추가하지 않았으며, 기존 flux 환산값인 `count_est`와 전처리·연산 시간 및 공통 알고리즘 설정은 추출 컬럼에서 제외했다. `raw_to_filtered_reduction_ratio`는 **raw / max(filtered, 1e-6)** 배율로, 제거율(%)과 다르다.

### 분석 산출물 목록

| 산출물 | 내용 |
| --- | --- |
| [optical_flow_jul_aug_analysis_data.xlsx](optical_flow_jul_aug_analysis_data.xlsx) | 배치 요약의 영상·영역·optical flow 핵심 컬럼 및 포함 여부 확인 |
| [batch_summary_coverage.json]({link('batch_summary_coverage.json')}) | 원본 CSV 해시, 파일 집합 대조, 월별 포함 현황, 엑셀 셀 대조 결과 |
| [report_reference_manifest.json]({link('report_reference_manifest.json')}) | 공유용 참고 파일의 원본·사본 경로, 크기와 SHA-256 |
| [batch_summary_coverage.csv]({link('tables/batch_summary_coverage.csv')}) | 원본 13,768개 영상별 배치·프레임·구간 결과·YAML·제외 기록 포함 여부 |
| [video_summary.csv]({link('tables/video_summary.csv')}) | 13,688개 영상별 활동량·방향 균형·변동성 |
| [frame_video_summary.csv]({link('tables/frame_video_summary.csv')}) | 전 프레임 후보 픽셀·component·flux 재집계 |
| [device.csv]({link('tables/device.csv')}) / [hour.csv]({link('tables/hour.csv')}) | 기기별 / 시각별 요약 |
| [device_month.csv]({link('tables/device_month.csv')}) / [month.csv]({link('tables/month.csv')}) | 기기·월 / 월별 요약 |
| [matched_device_hour_month.csv]({link('tables/matched_device_hour_month.csv')}) | 공통 323개 기기·시각의 월별 평균 |
| [matched_device_month.csv]({link('tables/matched_device_month.csv')}) | 공통 시각을 동일 비중으로 평균한 기기별 변화 |
| [normalized_day.csv]({link('tables/normalized_day.csv')}) | 완전한 기기·일의 날짜별 상대 활동량 |
| [complete_device_days.csv]({link('tables/complete_device_days.csv')}) | 17개 시각이 모두 있는 기기·일 |
| [normalized_hour.csv]({link('tables/normalized_hour.csv')}) | 기기 동일 비중 시간 곡선 및 공통 기기의 월별 시간 곡선 |
| [distribution.csv]({link('tables/distribution.csv')}) / [activity_quartiles.csv]({link('tables/activity_quartiles.csv')}) | 지표 분위수 / 활동량 사분위 그룹 |
| [top_videos.csv]({link('tables/top_videos.csv')}) / [top_windows.csv]({link('tables/top_windows.csv')}) | 큰 활동 영상 / 큰 3초 구간 |
| [summary.json]({link('summary.json')}) | 전체 요약, 상관계수, 합계 대조 결과 |
| [geometry_periods.csv]({link('tables/geometry_periods.csv')}) | 기기·기간별 실제 ROI/ENT 좌표, 경계 픽셀 수, 각 변의 면적 |
| [geometry_device_comparison.csv]({link('tables/geometry_device_comparison.csv')}) | 경계 전체 flux 및 픽셀 평균의 참고 정렬; 보정된 활동량 순위 아님 |
| [geometry_changes.csv]({link('tables/geometry_changes.csv')}) | 좌표 변경, 마스크 면적·위치 변화, 월별·근접 기간 flux 변화 |
| [geometry_month_groups.csv]({link('tables/geometry_month_groups.csv')}) | 동일 좌표 기기 및 변경 기기 그룹의 월별 비교 |
| [geometry_july_reference_month_cells.csv]({link('tables/geometry_july_reference_month_cells.csv')}) | 면적 비례 가정하의 기준 면적 환산 시나리오; 검증된 보정 아님 |
| [geometry_summary.json]({link('geometry_summary.json')}) | 공간 설정 확인 및 각 파생값의 해석 상태 |
| [normalization_review_cases.csv]({link('tables/normalization_review_cases.csv')}) | 합성 flow로 계산한 측정 의미의 반례; 실측 자료와 별도 |
| [normalization_review.json]({link('normalization_review.json')}) | 면적 정규화의 측정 의미 검토와 미검증 가정 |
| [figures]({link('figures')}) | 보고서에 사용한 그림 11개 |

재현 명령은 프로젝트 최상단에서 실행한다. 원본 영상의 optical flow를 다시 계산하지 않는다. 처음 두 단계는 실측 결과의 집계이며, 세 번째 단계는 별도의 합성 flow 개념 계산이다.

```bash
MPLCONFIGDIR=/tmp/bee_flow_mpl .venv/bin/python analysis/jul_aug_optical_flow/analyze.py
MPLCONFIGDIR=/tmp/bee_flow_mpl .venv/bin/python analysis/jul_aug_optical_flow/analyze_geometry.py
MPLCONFIGDIR=/tmp/bee_flow_mpl .venv/bin/python analysis/jul_aug_optical_flow/review_normalization_meaning.py
.venv/bin/python analysis/jul_aug_optical_flow/export_excel.py
MPLCONFIGDIR=/tmp/bee_flow_mpl .venv/bin/python analysis/jul_aug_optical_flow/write_report.py
```

분석 스크립트는 기존 프로젝트의 NumPy·Pandas·Matplotlib를 사용하며, 엑셀 작성은 OpenPyXL을 사용한다. Spearman 상관은 결측 쌍을 제외한 뒤 평균 순위의 Pearson 상관으로 계산한다. 0 신호가 아닌 영상의 CV에는 40개 구간의 모집단 표준편차를 사용한다.
""")

report = prepare_report_assets("\n\n".join(sections))
(ROOT / f"{REPORT_NAME}.md").write_text(report, encoding="utf-8")

# A small renderer for the Markdown constructs used in this report. Images are
# embedded so the HTML can be opened or printed without its supporting folder.
def inline(value):
    value = html.escape(value)
    value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)
    value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', value)
    return value

def render(text):
    lines = text.splitlines()
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i]); i += 1
            result.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
        elif re.match(r"^#{1,3} ", line):
            level = len(line.split(" ",1)[0])
            result.append(f"<h{level}>" + inline(line[level+1:]) + f"</h{level}>")
        elif line.startswith("!["):
            m = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line)
            assert m
            data = base64.b64encode((ROOT / m[2]).read_bytes()).decode()
            result.append(f'<img src="data:image/png;base64,{data}" alt="{html.escape(m[1],quote=True)}">')
        elif line.startswith("| "):
            rows = []
            while i < len(lines) and lines[i].startswith("| "):
                rows.append([v.strip() for v in lines[i].strip().strip("|").split("|")]); i += 1
            result.append('<div class="table-wrap"><table><thead><tr>' + "".join("<th>"+inline(v)+"</th>" for v in rows[0]) + "</tr></thead><tbody>")
            for row in rows[2:]:
                result.append("<tr>" + "".join("<td>"+inline(v)+"</td>" for v in row) + "</tr>")
            result.append("</tbody></table></div>")
            continue
        elif line.startswith("- ") or re.match(r"^\d+\. ",line):
            ordered = bool(re.match(r"^\d+\. ",line))
            tag = "ol" if ordered else "ul"
            result.append(f"<{tag}>")
            pattern = r"^\d+\. " if ordered else r"^- "
            while i < len(lines) and re.match(pattern,lines[i]):
                result.append("<li>"+inline(re.sub(pattern,"",lines[i]))+"</li>"); i += 1
            result.append(f"</{tag}>")
            continue
        else:
            paragraph = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||!\[|```|- |\d+\. )",lines[i]):
                paragraph.append(lines[i]); i += 1
            result.append("<p>" + "<br>".join(inline(p) for p in paragraph) + "</p>")
            continue
        i += 1
    return "\n".join(result)

document = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>2026년 7–8월 Optical Flow 결과 분석 보고서</title>
<style>
body{font-family:"Noto Sans CJK KR","Malgun Gothic",sans-serif;color:#193045;background:#edf2f5;margin:0;line-height:1.8}
main{max-width:1120px;margin:30px auto;padding:44px 52px;background:#fff;border-radius:10px}
h1{font-size:29px;line-height:1.5;margin:0 0 24px}h2{font-size:24px;margin-top:54px;border-bottom:2px solid #247ba0;padding-bottom:10px}
h3{font-size:19px;margin-top:30px}p,li{font-size:15px}strong{color:#125473}a{color:#166b94}
code{background:#f0f4f7;padding:2px 5px;border-radius:4px;font-size:13px}pre{overflow:auto;padding:18px;background:#f0f4f7;line-height:1.6}
pre code{padding:0}.table-wrap{overflow-x:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.65}
th,td{padding:9px 12px;border-bottom:1px solid #d7e0e6;text-align:left;white-space:nowrap}th{background:#e8f3f8;color:#164a63}
tbody tr:nth-child(even){background:#f7f9fb}img{width:100%;height:auto;margin-top:22px}ul,ol{padding-left:25px}li{margin:8px 0}
@media(max-width:720px){main{margin:0;padding:22px 18px}h1{font-size:24px}h2{font-size:21px}}
@media print{body{background:#fff}main{max-width:none;margin:0;padding:0}h2,h3{break-after:avoid}tr,img{break-inside:avoid}a{color:inherit}table{font-size:10px}th,td{padding:5px;white-space:normal}}
</style></head><body><main>""" + render(report) + "</main></body></html>"
(ROOT / f"{REPORT_NAME}.html").write_text(document, encoding="utf-8")
print(f"Wrote {REPORT_NAME}.md ({len(report):,} characters)")
print(f"Wrote {REPORT_NAME}.html ({len(document):,} characters; eleven embedded figures)")
