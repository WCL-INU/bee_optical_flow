"""Controlled counterexamples for interpreting area-normalized boundary flux.

These imposed flow fields exercise the repository's masks and filter stack.
They are conceptual checks, not estimates of bee counts or Farneback accuracy.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.bee_entrance_count import (Config, build_entrance_mask,
    build_counting_boundary_band, compute_raw_flux,
    update_persistence_filter, apply_component_area_filter)

shape = (280, 640)
config = Config(boundary_band_px=8, flow_mag_threshold=1.0,
    normal_flow_threshold=.5, use_persistence_filter=True,
    persist_decay=.65, persist_threshold=1.3,
    use_component_area_filter=True, min_flow_component_area=200)

def masks(rect):
    entrance = build_entrance_mask(shape, rect)
    return build_counting_boundary_band(entrance, rect, config)

def aggregate(rect, fields):
    band, nx, ny = masks(rect)
    persistence = np.zeros(shape, dtype=np.float32)
    raw_sum, filtered_sum, candidate_sum = 0., 0., 0
    for flow in fields:
        raw = compute_raw_flux(flow, band, nx, ny, config)
        persistent, persistence = update_persistence_filter(raw['candidate'], persistence, config)
        candidate, _ = apply_component_area_filter(persistent, config)
        raw_sum += raw['raw_in_flux'] + raw['raw_out_flux']
        filtered_sum += float(np.abs(raw['normal_flow'][candidate]).sum())
        candidate_sum += int(candidate.sum())
    duration = len(fields) / 24
    return dict(boundary_pixels=int(band.sum()), raw_flux=raw_sum,
                filtered_flux=filtered_sum, candidate_pixel_frames=candidate_sum,
                rate=filtered_sum/duration, density=filtered_sum/duration/band.sum(),
                duration_sec=duration)

def passages(xs):
    fields=[]
    # Each 24x24 patch moves down by two pixels each frame. All patches cross
    # only the top edge; none reach the entrance bottom edge in this sequence.
    for y in range(40,141,2):
        flow=np.zeros((*shape,2),dtype=np.float32)
        for x in xs:
            flow[y:y+24,x:x+24,1]=2
        fields.append(flow)
    return fields

narrow=(100,110,300,190)
wide=(100,110,580,190)
case_rows=[]
for case,rect,xs in [
    ('same_passage_narrow_boundary',narrow,[180]),
    ('same_passage_wide_boundary',wide,[180]),
    ('two_passages_wide_boundary',wide,[180,450])]:
    case_rows.append(dict(case=case,imposed_passage_patches=len(xs),**aggregate(rect,passages(xs))))

# The same localized, persistent flow is included by one boundary and excluded
# by a translated boundary of exactly the same area and shape.
localized=[]
for _ in range(20):
    flow=np.zeros((*shape,2),dtype=np.float32)
    flow[103:111,180:212,1]=2
    localized.append(flow)
for case,rect in [('localized_signal_included',narrow),
                  ('localized_signal_excluded_same_area',(100,140,300,220))]:
    case_rows.append(dict(case=case,imposed_passage_patches=np.nan,**aggregate(rect,localized)))

cases=pd.DataFrame(case_rows)
assert cases.iloc[0].filtered_flux>0
assert cases.iloc[0].filtered_flux==cases.iloc[1].filtered_flux
assert cases.iloc[2].filtered_flux==2*cases.iloc[0].filtered_flux
assert cases.iloc[3].boundary_pixels==cases.iloc[4].boundary_pixels
assert cases.iloc[3].filtered_flux>0 and cases.iloc[4].filtered_flux==0
cases.to_csv(HERE/'tables/normalization_review_cases.csv',index=False)
review=dict(
    purpose='conceptual counterexamples using imposed flow, not observations of real bees',
    status='area_normalization_not_validated_as_activity_correction',
    assumptions_for_reference_area_scaling=[
        'Expected flux per unit of included boundary area is exchangeable',
        'Boundary changes primarily add or remove comparable signal-bearing area',
        'Position, projection, pixel scale, and filter effects do not change the relationship'],
    unchanged_passage_flux_ratio=float(cases.iloc[1].filtered_flux/cases.iloc[0].filtered_flux),
    unchanged_passage_density_ratio=float(cases.iloc[1].density/cases.iloc[0].density),
    doubled_passages_flux_ratio=float(cases.iloc[2].filtered_flux/cases.iloc[0].filtered_flux),
    doubled_passages_density_ratio=float(cases.iloc[2].density/cases.iloc[0].density),
    equal_area_translated_signal_flux_ratio=float(cases.iloc[4].filtered_flux/cases.iloc[3].filtered_flux))
(HERE/'normalization_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2))

font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family']=font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams['axes.unicode_minus']=False
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
labels=['좁은 경계\n통과 패치 1개','넓은 경계\n같은 패치 1개','넓은 경계\n동일 패치 2개']
q=cases.iloc[:3]
axes[0].bar(labels,q.filtered_flux/q.iloc[0].filtered_flux,color=['#247BA0','#247BA0','#E59E36'])
axes[0].set(title='동일 합산·필터 규칙의 누적 flux',ylabel='첫 사례 대비 배율',ylim=(0,2.2))
axes[1].bar(labels,q.density/q.iloc[0].density,color=['#247BA0','#247BA0','#E59E36'])
axes[1].set(title='전체 경계 면적으로 나눈 값',ylabel='첫 사례 대비 배율',ylim=(0,2.2))
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('개념 검증: 동일한 이동 패치의 수와 경계 크기를 독립적으로 바꾼 합성 flow')
fig.tight_layout()
fig.savefig(HERE/'figures/11_normalization_meaning_review.png',dpi=180)
plt.close(fig)
print(cases.to_string(index=False))
print(json.dumps(review,ensure_ascii=False,indent=2))
