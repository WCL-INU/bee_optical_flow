"""Describe geometry and flux; area quotients are not calibrated corrections.

Dividing by mask area changes the measured quantity to a spatial mean. Scaling
to a reference area is only a scenario under an unverified proportionality
assumption, not an estimate of the causal effect of changing the mask.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.bee_entrance_count import Config, build_entrance_mask, build_counting_boundary_band

HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
FIGURES = HERE / "figures"
SOURCE = ROOT / "bee_count_output/yaml_jul_aug_all"
COORDS = ["roi_x1", "roi_y1", "roi_x2", "roi_y2", "ent_x1", "ent_y1", "ent_x2", "ent_y2", "boundary_band_px"]
d = pd.read_csv(TABLES / "video_summary.csv", parse_dates=["date"])
original = pd.read_csv(SOURCE / "batch_summary.csv")
assert set(original.video) == set(d.video)
summary_rows = original.set_index("video")
regions = yaml.safe_load((ROOT / "roi_regions.yaml").read_text())["regions"]
for r in regions:
    r["start_stamp"] = pd.Timestamp(r["start"])
    r["end_stamp"] = pd.Timestamp(r["end"])
geometry_cache = {}
mask_cache = {}

def identity(values):
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()[:12]

def build_geometry(values, size):
    rx, ry, rx2, ry2, x, y, x2, y2, band_width = values
    local = (x-rx, y-ry, x2-rx, y2-ry)
    entrance = build_entrance_mask((ry2-ry, rx2-rx), local)
    band, nx, ny = build_counting_boundary_band(entrance, local, Config(boundary_band_px=band_width))
    full = np.zeros((size[1],size[0]), dtype=bool)
    full[ry:ry2,rx:rx2] = band
    g = dict(zip(COORDS,values))
    g.update(geometry_id=identity(values), ent_width=x2-x, ent_height=y2-y,
             ent_area=(x2-x)*(y2-y), perimeter=2*(x2-x+y2-y),
             roi_area=(rx2-rx)*(ry2-ry), boundary_pixels=int(band.sum()),
             top_pixels=int((band & (ny==1)).sum()), bottom_pixels=int((band & (ny==-1)).sum()),
             left_pixels=int((band & (nx==1)).sum()), right_pixels=int((band & (nx==-1)).sum()))
    assert g["boundary_pixels"] == sum(g[k] for k in ["top_pixels","bottom_pixels","left_pixels","right_pixels"])
    return g,full

def read_config(row):
    stem = Path(row.video).stem
    device_name, day, clock = stem.rsplit("_",2)
    stamp = pd.Timestamp(f"{day[:4]}-{day[4:6]}-{day[6:]} {clock[:2]}:{clock[2:4]}:{clock[4:]}")
    matches = [r for r in regions if r["device"]==device_name and r["start_stamp"]<=stamp<=r["end_stamp"]]
    assert len(matches)==1,(row.video,len(matches))
    r = matches[0]
    yaml_values = list(r["roi"])+list(r["entrance"])+[8]
    marker = SOURCE / f"{stem}_completion.json"
    if marker.exists():
        config = json.loads(marker.read_text())["signature"]["config"]
        values = [int(config[k]) for k in COORDS]
        source = "completion_config"
    elif summary_rows.loc[row.video,COORDS].notna().all():
        values = [int(v) for v in summary_rows.loc[row.video,COORDS]]
        source = "summary_config"
    else:
        values = yaml_values
        source = "yaml_period_reconstruction"
    return dict(video=row.video, config_source=source, values=values, yaml_values=yaml_values,
                size=r["image_size"], yaml_matches_stored=values==yaml_values,
                yaml_start=r["start"],yaml_end=r["end"])

print("Reading saved geometry configurations",flush=True)
rows=[]
with ThreadPoolExecutor(max_workers=4) as pool:
    for i,r in enumerate(pool.map(read_config,d.itertuples(index=False)),1):
        key=tuple(r.pop("values"))
        r.pop("yaml_values")
        size=r.pop("size")
        if key not in geometry_cache:
            geometry_cache[key],mask_cache[key]=build_geometry(list(key),size)
        rows.append(r | geometry_cache[key])
        if i%4000==0:print(f"Geometry {i:,}/{len(d):,}",flush=True)
d=d.merge(pd.DataFrame(rows),on="video",validate="one_to_one")
assert d.yaml_matches_stored.all()
d["density"]=d.rate/d.boundary_pixels
d["perimeter_density"]=d.rate/d.perimeter
reference_bands=d[d.month==7].groupby("device").boundary_pixels.first()
d["july_reference_band"]=d.device.map(reference_bands)
d["july_area_adjusted_rate"]=d.rate*d.july_reference_band/d.boundary_pixels
f=pd.read_csv(TABLES / "frame_video_summary.csv")
d=d.merge(f[["video","frame_n","frame_sum_raw_candidate_pixels","frame_sum_filtered_candidate_pixels"]],on="video",validate="one_to_one")
d["raw_occupancy"]=d.frame_sum_raw_candidate_pixels/(d.frame_n*d.boundary_pixels)
d["filtered_occupancy"]=d.frame_sum_filtered_candidate_pixels/(d.frame_n*d.boundary_pixels)
assert d.raw_occupancy.max()<=1
assert (d.filtered_occupancy<=d.raw_occupancy+1e-12).all()
d.to_csv(TABLES / "geometry_video_summary.csv",index=False)
periods=d.groupby(["device","geometry_id"],as_index=False).agg(
    first=("date","min"),last=("date","max"),n=("video","size"),
    **{c:(c,"first") for c in COORDS+["ent_width","ent_height","ent_area","perimeter","roi_area","boundary_pixels",
                                      "top_pixels","bottom_pixels","left_pixels","right_pixels"]},
    mean_rate=("rate","mean"),mean_density=("density","mean"))
periods=periods.sort_values(["device","first"])
periods.to_csv(TABLES / "geometry_periods.csv",index=False)
device=d.groupby("device").agg(n=("video","size"),geometry_n=("geometry_id","nunique"),
    band_min=("boundary_pixels","min"),band_max=("boundary_pixels","max"),
    mean_band=("boundary_pixels","mean"),mean_rate=("rate","mean"),median_rate=("rate","median"),
    mean_density=("density","mean"),median_density=("density","median"),
    mean_perimeter_density=("perimeter_density","mean"),mean_occupancy=("filtered_occupancy","mean"))
device["raw_rank"]=device.mean_rate.rank(ascending=False,method="min").astype(int)
device["density_rank"]=device.mean_density.rank(ascending=False,method="min").astype(int)
device["rank_change"]=device.raw_rank-device.density_rank
device.reset_index().to_csv(TABLES / "geometry_device_comparison.csv",index=False)
changed=device.index[device.geometry_n>1].tolist()
stable=device.index[device.geometry_n==1].tolist()

def paired_month(df,keys):
    cell=df.groupby(keys+["month"])[["rate","density","boundary_pixels"]].mean().unstack("month")
    cell=cell.dropna(subset=[("rate",7),("rate",8)])
    out=pd.DataFrame(index=cell.index)
    for col in ["rate","density","boundary_pixels"]:
        out[col+"_july"]=cell[(col,7)]
        out[col+"_august"]=cell[(col,8)]
        out[col+"_change"]=out[col+"_august"]/out[col+"_july"].replace(0,np.nan)-1
    return out.reset_index()

monthly_cells=paired_month(d,["device","hour"])
monthly_cells.to_csv(TABLES / "geometry_matched_month_cells.csv",index=False)
monthly=monthly_cells.groupby("device").agg(cells=("hour","size"),
    rate_july=("rate_july","mean"),rate_august=("rate_august","mean"),
    density_july=("density_july","mean"),density_august=("density_august","mean"),
    band_july=("boundary_pixels_july","mean"),band_august=("boundary_pixels_august","mean"))
monthly["raw_change"]=monthly.rate_august/monthly.rate_july-1
monthly["density_change"]=monthly.density_august/monthly.density_july-1
monthly["band_change"]=monthly.band_august/monthly.band_july-1
monthly["geometry_changed"]=monthly.index.isin(changed)
monthly.reset_index().to_csv(TABLES / "geometry_device_month.csv",index=False)
same_geometry=paired_month(d,["device","hour","geometry_id"])
same_geometry.to_csv(TABLES / "geometry_same_coordinates_month_cells.csv",index=False)
group_rows=[]
for name,df in [("all_common_devices",monthly_cells),
                ("unchanged_devices",monthly_cells[monthly_cells.device.isin(stable)]),
                ("changed_devices",monthly_cells[monthly_cells.device.isin(changed)]),
                ("exact_same_geometry",same_geometry)]:
    row=dict(group=name,devices=df.device.nunique(),cells=len(df))
    for metric in ["rate","density"]:
        row[metric+"_july"]=df[metric+"_july"].mean()
        row[metric+"_august"]=df[metric+"_august"].mean()
        row[metric+"_change"]=row[metric+"_august"]/row[metric+"_july"]-1
    group_rows.append(row)
groups=pd.DataFrame(group_rows)
groups.to_csv(TABLES / "geometry_month_groups.csv",index=False)
reference_cells=d.groupby(["device","hour","month"])["july_area_adjusted_rate"].mean().unstack("month").dropna()
reference_cells.columns=["july_area_adjusted_july","july_area_adjusted_august"]
reference_cells.reset_index().to_csv(TABLES / "geometry_july_reference_month_cells.csv",index=False)
reference_july=float(reference_cells.july_area_adjusted_july.mean())
reference_august=float(reference_cells.july_area_adjusted_august.mean())

changes=[]
for dev in changed:
    p=periods[periods.device==dev]
    assert len(p)==2
    a,b=p.iloc[0],p.iloc[1]
    old_key=tuple(int(a[c]) for c in COORDS)
    new_key=tuple(int(b[c]) for c in COORDS)
    old,new=mask_cache[old_key],mask_cache[new_key]
    center_old=np.array([(a.ent_x1+a.ent_x2)/2,(a.ent_y1+a.ent_y2)/2])
    center_new=np.array([(b.ent_x1+b.ent_x2)/2,(b.ent_y1+b.ent_y2)/2])
    row=dict(device=dev,last_old_recording=str(a['last'].date()),first_new_recording=str(b['first'].date()),
             old_band=int(a.boundary_pixels),new_band=int(b.boundary_pixels),
             band_change=b.boundary_pixels/a.boundary_pixels-1,
             old_ent_area=int(a.ent_area),new_ent_area=int(b.ent_area),ent_area_change=b.ent_area/a.ent_area-1,
             old_perimeter=int(a.perimeter),new_perimeter=int(b.perimeter),
             center_dx=float(center_new[0]-center_old[0]),center_dy=float(center_new[1]-center_old[1]),
             center_shift=float(np.linalg.norm(center_new-center_old)),
             band_iou=float((old&new).sum()/(old|new).sum()),
             month_raw_change=float(monthly.loc[dev,"raw_change"]),
             month_density_change=float(monthly.loc[dev,"density_change"]))
    before=d[(d.device==dev)&d.date.between("2026-08-03","2026-08-09")]
    after=d[(d.device==dev)&d.date.between("2026-08-11","2026-08-17")]
    hourly=pd.concat([before.assign(side="before"),after.assign(side="after")]).groupby(["hour","side"])[["rate","density"]].mean().unstack("side")
    if set(hourly.columns.get_level_values(1))=={"before","after"}:
        hourly=hourly.dropna()
        row.update(local_common_hours=len(hourly),before_n=len(before),after_n=len(after),
            before_days=before.date.nunique(),after_days=after.date.nunique(),
            local_raw_change=float(hourly["rate"]["after"].mean()/hourly["rate"]["before"].mean()-1),
            local_density_change=float(hourly["density"]["after"].mean()/hourly["density"]["before"].mean()-1))
    else:
        row.update(local_common_hours=0,before_n=len(before),after_n=len(after),before_days=before.date.nunique(),after_days=after.date.nunique(),
                   local_raw_change=np.nan,local_density_change=np.nan)
    changes.append(row)
changes=pd.DataFrame(changes)
changes.to_csv(TABLES / "geometry_changes.csv",index=False)

# Unchanged-coordinate devices observed in both neighboring calendar periods.
local=d[d.date.between("2026-08-03","2026-08-09")|d.date.between("2026-08-11","2026-08-17")].copy()
local["side"]=np.where(local.date<"2026-08-11","before","after")
local_cells=local.groupby(["device","hour","side"])[["rate","density"]].mean().unstack("side").dropna()
local_rows=[]
for dev,cell in local_cells.groupby(level="device"):
    local_rows.append(dict(device=dev,hours=len(cell),geometry_changed=dev in changed,
        raw_before=cell[("rate","before")].mean(),raw_after=cell[("rate","after")].mean(),
        density_before=cell[("density","before")].mean(),density_after=cell[("density","after")].mean()))
local_device=pd.DataFrame(local_rows)
local_device["raw_change"]=local_device.raw_after/local_device.raw_before-1
local_device["density_change"]=local_device.density_after/local_device.density_before-1
local_device.to_csv(TABLES / "geometry_local_period_comparison.csv",index=False)
control=local_device[~local_device.geometry_changed]
control_raw_change=control.raw_after.mean()/control.raw_before.mean()-1
control_density_change=control.density_after.mean()/control.density_before.mean()-1

summary=dict(changed_devices=changed,stable_devices=stable,geometry_n=len(periods),
    interpretation=dict(
        density='Spatial mean over all boundary pixels, including zero-signal pixels; not corrected total activity',
        density_rank='Ordering of a different spatial statistic, not a validated correction to activity ranking',
        july_reference_area_change='Conditional sensitivity scenario assuming flux proportional to boundary area; not an empirically validated correction',
        geometry_changes='Observed coordinate and signal changes; area differences alone do not identify their causal contributions'),
    config_sources=d.config_source.value_counts().to_dict(),stored_yaml_mismatch=int((~d.yaml_matches_stored).sum()),
    boundary_min=int(d.boundary_pixels.min()),boundary_max=int(d.boundary_pixels.max()),
    area_ratio=float(d.boundary_pixels.max()/d.boundary_pixels.min()),
    july_reference_area_july_rate=reference_july,july_reference_area_august_rate=reference_august,
    july_reference_area_change=reference_august/reference_july-1,
    raw_density_rank_spearman=float(device.raw_rank.corr(device.density_rank)),
    local_control_devices=control.device.tolist(),local_control_raw_change=float(control_raw_change),
    local_control_density_change=float(control_density_change),
    monthly_groups=groups.to_dict(orient="records"),changes=changes.to_dict(orient="records"),
    density_top_devices=device.sort_values("density_rank").head(5).index.tolist(),
    density_last_devices=device.sort_values("density_rank").tail(3).index.tolist())
def valid_json(value):
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {k:valid_json(v) for k,v in value.items()}
    if isinstance(value,list):return [valid_json(v) for v in value]
    return value
summary=valid_json(summary)
(HERE / "geometry_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

font_path="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"]=font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"]=False
blue,orange="#247BA0","#E59E36"
fig,ax=plt.subplots(1,2,figsize=(12,6))
order=device.sort_values("mean_rate")
ax[0].barh(order.index.astype(str),order.mean_rate/1000,color=blue)
ax[0].set(title="경계 전체의 누적 신호",xlabel="천 flux / 초",ylabel="기기")
order=device.sort_values("mean_density")
ax[1].barh(order.index.astype(str),order.mean_density,color=orange)
ax[1].set(title="전체 경계 픽셀의 평균 신호 (참고 지표)",xlabel="flux / (초·경계 픽셀)",ylabel="기기")
fig.tight_layout();fig.savefig(FIGURES / "08_geometry_normalized_devices.png",dpi=180);plt.close(fig)

fig,axes=plt.subplots(3,2,figsize=(13,9))
for ax,dev in zip(axes.flat,changed):
    p=periods[periods.device==dev]
    a,b=p.iloc[0],p.iloc[1]
    for row,color,label in [(a,blue,"변경 전"),(b,orange,"변경 후")]:
        # Draw actual band coordinates, retaining image-space positions.
        key=tuple(int(row[c]) for c in COORDS)
        yy,xx=np.nonzero(mask_cache[key])
        ax.scatter(xx[::5],yy[::5],s=.3,color=color,alpha=.7,label=label,rasterized=True)
    ax.set(title=f"{dev}번: {int(a.boundary_pixels):,} → {int(b.boundary_pixels):,} 픽셀",
           xlim=(0,1640),ylim=(1232,880),xlabel="영상 x 좌표",ylabel="영상 y 좌표")
    ax.legend(markerscale=8,fontsize=8)
fig.tight_layout();fig.savefig(FIGURES / "09_boundary_coordinate_changes.png",dpi=180);plt.close(fig)

daily=d.groupby(["device","date","geometry_id"],as_index=False).agg(density=("density","mean"),rate=("rate","mean"))
daily.to_csv(TABLES / "geometry_daily_activity.csv",index=False)
fig,axes=plt.subplots(3,2,figsize=(13,9),sharex=True)
for ax,dev in zip(axes.flat,changed):
    q=daily[daily.device==dev]
    for gid,part in q.groupby("geometry_id"):
        color=blue if part.date.min()<pd.Timestamp("2026-08-11") else orange
        ax.plot(part.date,part.density,"o-",markersize=2,linewidth=1,color=color)
    ax.axvline(pd.Timestamp("2026-08-11"),color="gray",linestyle="--",linewidth=1)
    ax.set(title=f"{dev}번 기기",ylabel="flux / (초·경계 픽셀)")
fig.autofmt_xdate();fig.tight_layout();fig.savefig(FIGURES / "10_density_by_geometry_period.png",dpi=180);plt.close(fig)
print("Geometry tables and figures written",flush=True)
