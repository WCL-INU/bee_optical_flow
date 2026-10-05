# Bee Entrance Optical Flow 프로젝트 이해 개요

대상: optical flow, 영상 처리, 벌 출입량 추정 방법론을 처음 접하는 팀원  
목적: 현재 디렉토리의 작업이 무엇을 하려는지, 어떤 지식이 필요하며, 코드를 어떤 순서로 따라가면 되는지 정리한다.

---

## 1. 한 줄 요약

이 프로젝트는 벌통 입구 영상에서 벌 한 마리씩을 탐지하거나 추적하지 않고, 입구 경계 주변의 움직임 방향과 크기를 optical flow로 계산해 `IN`/`OUT` 활동량을 3초 단위로 추정하는 실험이다.

---

## 2. 현재 진행 중인 작업

현재 작업의 중심은 `src/bee_entrance_count.py`이다.

1. 영상에서 벌통 입구 주변 ROI만 잘라낸다.
2. ROI 안에 고정된 입구 사각형을 둔다.
3. 입구의 top, bottom, left, right 네 경계 주변을 counting boundary로 사용한다.
4. 연속 프레임 사이의 Farneback optical flow를 계산한다.
5. 움직임 벡터를 입구 경계의 normal 방향으로 투영해 `IN`/`OUT` flux를 계산한다.
6. Gaussian blur, temporal persistence filter, connected-component area filter로 노이즈를 줄인다.
7. 프레임별 flux CSV, 3초 window 요약 CSV, preview video를 만든다.

이전에는 event detector 방식도 문서화되어 있지만, 현재 구현은 먼저 optical-flow 신호 자체를 안정화하는 방향으로 돌아와 있다.

---

## 3. 프로젝트의 목표와 비목표

### 목표

- 벌통 입구 주변의 출입 활동량을 자동으로 추정한다.
- 완벽한 개체 수가 아니라, 시간 구간별 활동량과 방향성을 안정적으로 얻는다.
- 여러 영상과 파라미터 preset을 비교해 노이즈에 강한 설정을 찾는다.

### 비목표

- YOLO 같은 객체 탐지 모델을 학습하는 프로젝트가 아니다.
- 각 벌에게 ID를 붙여 tracking하는 프로젝트가 아니다.
- 현재 단계에서 절대적인 실제 벌 마릿수를 완벽히 맞추는 것이 1순위는 아니다.

---

## 4. 폴더와 파일 지도

| 경로 | 역할 |
| --- | --- |
| `src/bee_entrance_count.py` | 핵심 알고리즘. optical flow, flux 계산, 필터링, CSV/video 출력 |
| `src/main.py` | batch, compare, groups, evaluate, tune 실행 모드 제공 |
| `src/optical_count.py` | 기본 비교 영상 2개를 실행하는 간단한 wrapper |
| `src/vis.py` | optical-flow magnitude heatmap preview 실험 |
| `src/vis_arr.py` | flow vector를 화살표로 보는 실험 |
| `src/vis_arr2.py` | 방향을 HSV 색상으로 보는 실험 |
| `src/sep_blob.py` | 움직임 blob/component 분리 실험 |
| `docs/` | event counting, noise filtering, persistence filter 판단 기록 |
| `videos/` | 로컬 원본 영상과 일부 preview 영상 |
| `bee_count_output/` | 실행 결과 CSV와 preview video |

---

## 5. 먼저 알아야 할 영상 처리 기본 개념

### Frame

영상은 여러 장의 이미지가 시간 순서대로 이어진 것이다. 각 이미지를 frame이라고 한다.

### Pixel

이미지를 이루는 가장 작은 점이다. 이 프로젝트는 각 pixel 또는 pixel 영역의 움직임을 계산한다.

### ROI

Region of Interest의 줄임말이다. 전체 영상에서 분석에 필요한 벌통 입구 주변 영역만 잘라낸 부분이다.

현재 코드 기본값:

```python
roi_x1, roi_y1, roi_x2, roi_y2 = 1300, 1000, 1640, 1232
```

### Entrance Rectangle

ROI 안에서 실제 벌통 입구로 보는 사각형이다.

현재 코드 기본값:

```python
ent_x1, ent_y1, ent_x2, ent_y2 = 1400, 1200, 1600, 1232
```

주의: `roi_ent_info.txt`에는 다른 좌표 메모가 있다. 실제 실험 영상마다 좌표 검증이 필요하다.

---

## 6. Optical Flow란?

Optical flow는 이전 frame에서 다음 frame으로 넘어갈 때 이미지의 각 지점이 어느 방향으로 얼마나 움직였는지를 추정하는 방법이다.

이 프로젝트에서는 OpenCV의 Farneback optical flow를 사용한다.

결과는 각 pixel마다 다음과 같은 2차원 벡터다.

```text
flow[y, x] = (dx, dy)
```

- `dx`: 좌우 방향 이동량
- `dy`: 상하 방향 이동량
- `magnitude`: 움직임의 크기
- `direction`: 움직임의 방향

벌이 이동하면 입구 주변 pixel texture가 변하고, optical flow가 그 움직임을 벡터로 표현한다.

---

## 7. 왜 개체 탐지 대신 optical flow를 쓰는가?

벌통 입구에서는 벌들이 겹치고, 빠르게 움직이고, 일부만 보이고, 그림자나 날개 움직임도 생긴다. 이런 상황에서 개별 벌을 안정적으로 탐지하고 ID tracking하는 것은 어렵다.

대신 입구 경계를 지나가는 전체 움직임의 양을 재면, 개별 ID 없이도 활동량을 추정할 수 있다.

이 프로젝트의 핵심 질문은 다음과 같다.

```text
입구 경계 방향으로 들어가는 움직임이 얼마나 있었는가?
입구 경계 방향으로 나오는 움직임이 얼마나 있었는가?
```

---

## 8. Boundary와 Normal Vector

입구 사각형 전체가 아니라, 입구의 경계 주변 얇은 띠만 counting 영역으로 사용한다.

현재는 top, bottom, left, right 네 경계를 모두 사용한다. bottom 경계가 ROI 하단과 맞닿아 있더라도 같은 4방향 누적 규칙으로 optical-flow flux를 합산한다.

Normal vector는 경계에 수직인 방향 벡터다. 움직임 벡터를 이 normal 방향으로 투영하면, 움직임이 입구 안쪽으로 향하는지 바깥쪽으로 향하는지 판단할 수 있다.

```text
positive normal flow  -> IN
negative normal flow  -> OUT
```

카메라 방향이나 좌표 설정이 다르면 생물학적 의미의 IN/OUT이 뒤집힐 수 있으므로 preview로 확인해야 한다.

---

## 9. Flux란?

Flux는 경계를 통과하는 움직임의 총량이다.

이 프로젝트에서는 candidate pixel들의 normal 방향 움직임을 합산한다.

```text
IN flux  = normal 방향으로 입구 안쪽을 향하는 움직임의 합
OUT flux = normal 방향으로 입구 바깥쪽을 향하는 움직임의 합
```

개별 벌을 세는 것이 아니라, 경계를 통과하는 움직임의 에너지를 세는 방식에 가깝다.

---

## 10. 전체 처리 흐름

```text
input video
  -> crop ROI
  -> grayscale 변환
  -> Gaussian blur
  -> Farneback optical flow
  -> entrance boundary band 생성
  -> flow를 boundary normal 방향으로 투영
  -> raw candidate pixel 선택
  -> persistence filter
  -> optional component area filter
  -> frame별 IN/OUT flux
  -> 3초 window 요약
  -> CSV + preview video 저장
```

코드에서 이 흐름은 `process_video()`가 담당한다.

---

## 11. Raw Candidate Pixel

모든 pixel의 optical flow를 쓰면 노이즈가 너무 많다. 그래서 다음 조건을 통과한 pixel만 raw candidate가 된다.

```python
candidate = (
    counting_boundary_band
    & (mag > flow_mag_threshold)
    & (abs(normal_flow) > normal_flow_threshold)
)
```

의미:

- 입구 경계 주변에 있어야 한다.
- 움직임 크기가 충분히 커야 한다.
- 입구 경계 normal 방향 움직임이 충분히 커야 한다.

---

## 12. 노이즈가 생기는 이유

Optical flow는 벌의 실제 움직임뿐 아니라 다음 변화에도 반응한다.

- 영상 압축 artifact
- 조명 변화
- 카메라 미세 흔들림
- 배경 texture 변화
- 벌 날개나 그림자처럼 실제 출입과 직접 관련 없는 움직임

그래서 raw flux를 그대로 count로 바꾸면 조용한 영상도 활동량이 큰 것처럼 보일 수 있다.

---

## 13. Gaussian Blur

Gaussian blur는 optical flow 계산 전에 grayscale frame을 살짝 흐리게 만든다.

목적:

- 작은 pixel 단위 노이즈를 줄인다.
- 압축 artifact나 미세 texture 변화에 덜 민감하게 만든다.

현재 사용 가능한 kernel:

```text
1: blur 없음
3: 기본 blur
5: 강한 blur
```

너무 강한 blur는 작은 벌 움직임까지 없앨 수 있다.

---

## 14. Temporal Persistence Filter

한 frame에만 갑자기 튀는 움직임은 노이즈일 가능성이 높다. Persistence filter는 candidate가 여러 frame 동안 지속되는지 확인한다.

개념:

```python
persistence = persist_decay * persistence + candidate
persistent_candidate = persistence > persist_threshold
filtered_candidate = candidate & persistent_candidate
```

효과:

- 짧은 spike를 제거한다.
- 시간적으로 이어지는 움직임은 통과시킨다.

핵심 파라미터:

| 파라미터 | 의미 |
| --- | --- |
| `persist_decay` | 이전 frame의 흔적을 얼마나 오래 유지할지 |
| `persist_threshold` | 몇 frame 정도 지속되어야 통과시킬지 |

---

## 15. Connected Component Area Filter

Candidate mask에서 서로 연결된 pixel 덩어리를 component라고 한다.

Area filter는 작은 component를 제거한다.

```text
작고 흩어진 점들 -> 노이즈일 가능성 높음
어느 정도 면적을 가진 덩어리 -> 벌 움직임일 가능성 높음
```

현재 `src/main.py`의 `selected` preset은 area filter를 사용한다.

```python
use_component_area_filter=True
min_flow_component_area=200
```

너무 큰 최소 면적을 요구하면 실제 작은 벌 움직임도 제거될 수 있다.

---

## 16. Optional Filter

현재 구현에는 기본 필터 외에 선택적으로 켤 수 있는 필터가 있다.

| 옵션 | 목적 | 위험 |
| --- | --- | --- |
| `use_global_flow_compensation` | ROI 배경 전체가 같은 방향으로 흔들리는 motion 제거 | 벌이 많으면 배경 추정이 오염될 수 있음 |
| `use_bidirectional_balance_filter` | IN/OUT이 비슷하게 동시에 생기는 진동성 노이즈 억제 | 실제 동시 출입을 줄여버릴 수 있음 |

기본값은 꺼져 있다.

---

## 17. Count Estimate의 의미

3초 window별 count estimate는 flux를 임의의 단위값으로 나눈 값이다.

```python
filtered_in_count_est = filtered_in_flux_sum / in_bee_flux_unit
filtered_out_count_est = filtered_out_flux_sum / out_bee_flux_unit
```

기본 `in_bee_flux_unit`, `out_bee_flux_unit`은 `100.0`이다.

중요: 이 값은 calibration 전에는 실제 벌 마릿수로 해석하면 안 된다. 사람이 직접 센 truth label을 이용해 flux unit을 보정해야 한다.

---

## 18. 현재 주요 파라미터

`src/bee_entrance_count.py`의 `Config` 기본값:

| 파라미터 | 기본값 | 의미 |
| --- | ---: | --- |
| `boundary_band_px` | `8` | 입구 경계 주변 counting band 폭 |
| `flow_mag_threshold` | `0.30` | 움직임 크기 최소값 |
| `normal_flow_threshold` | `0.08` | normal 방향 움직임 최소값 |
| `preview_panel_width` | `360` | preview 영상 오른쪽 정보 패널 폭 |
| `blur_kernel` | `3` | blur 강도 |
| `use_persistence_filter` | `True` | persistence filter 사용 |
| `persist_decay` | `0.65` | persistence 감쇠율 |
| `persist_threshold` | `1.3` | persistence 통과 기준 |
| `use_component_area_filter` | `False` | component area filter 사용 여부 |
| `min_flow_component_area` | `30` | component 최소 면적 |
| `window_sec` | `3.0` | 요약 window 길이 |

`src/main.py`의 기본 preset은 `selected`이며, 더 강한 필터링을 사용한다.

```python
blur_kernel=5
flow_mag_threshold=1.0
normal_flow_threshold=0.5
use_component_area_filter=True
min_flow_component_area=200
```

---

## 19. 실행 모드

`src/main.py`는 여러 실험을 쉽게 반복하기 위한 runner다.

| 모드 | 설명 |
| --- | --- |
| `batch` | 선택된 영상들을 각각 처리하고 `batch_summary.csv` 생성 |
| `compare` | 여러 영상을 같은 설정으로 처리하고 비교 |
| `groups` | sliding window 방식으로 영상 그룹 비교 |
| `evaluate` | summary CSV와 truth CSV를 비교해 평가 |
| `tune` | 여러 파라미터 조합을 돌려 평가 점수 비교 |

예시:

```bash
uv run python -m src.main --mode batch --preset selected --dry-run
uv run python -m src.main --mode batch --preset selected
uv run python -m src.main --mode tune --preset selected --truth-csv videos/entrance.csv
```

---

## 20. 출력 파일

단일 영상 처리 시 주요 출력:

| 파일 | 의미 |
| --- | --- |
| `{video_stem}_preview.mp4` | ROI, 입구 사각형, 4방향 counting band, 후보 flow, 별도 정보 패널을 시각화한 영상 |
| `{video_stem}_frame_flux.csv` | 프레임별 raw/filtered IN/OUT flux |
| `{video_stem}_window_3sec.csv` | 3초 window별 flux와 count estimate |

batch/compare/tune/evaluate 실행 시 추가 출력:

| 파일 | 의미 |
| --- | --- |
| `batch_summary.csv` | 영상별 전체 flux 요약 |
| `comparison_summary.csv` | 비교 실행 결과 요약 |
| `evaluation.csv` | truth와 prediction 비교 |
| `evaluation_metrics.csv` | MAE, score 등 평가 지표 |
| `tuning_results.csv` | 파라미터 조합별 평가 결과 |

---

## 21. 현재 로컬 상태 요약

현재 디렉토리에서 확인한 상태:

- 자료 작성 전 기준으로 기존 Git 변경 사항은 없었고, 이 문서 `presentation.md`가 새로 추가되었다.
- `videos/`에는 `ANU-25-summer-*20260328_130000.mp4` 계열 영상들이 있다.
- `bee_count_output/persistence_filter/`에는 `ANU-25-summer-11_20260328_130000` 영상에 대한 preview, frame CSV, window CSV가 있다.
- `videos/entrance.csv` 같은 truth label 파일은 현재 파일 목록에서 확인되지 않았다. `evaluate`와 `tune`을 제대로 쓰려면 사람이 센 정답 CSV가 필요하다.

---

## 22. 팀원이 코드를 따라가는 추천 순서

1. `README.md`를 읽고 프로젝트의 목표와 현재 방향을 파악한다.
2. `docs/bee_entrance_persistence_filter.md`를 읽고 현재 필터링 전략을 이해한다.
3. `src/bee_entrance_count.py`의 `Config`를 본다.
4. `process_video()`를 따라가며 전체 pipeline을 확인한다.
5. `compute_raw_flux()`에서 flux 계산 방식을 확인한다.
6. `update_persistence_filter()`와 `apply_component_area_filter()`에서 노이즈 제거 방식을 확인한다.
7. `aggregate_window_counts()`에서 3초 요약이 어떻게 만들어지는지 본다.
8. `src/main.py`의 `PRESETS`와 실행 모드를 확인한다.
9. preview video와 CSV를 같이 보며 파라미터가 어떤 효과를 내는지 확인한다.

---

## 23. 검증할 때 봐야 하는 것

Preview video에서 확인할 것:

- ROI가 실제 벌통 입구 주변을 잘 포함하는가?
- 빨간 입구 사각형이 실제 입구와 맞는가?
- cyan counting band가 top, bottom, left, right 네 경계에 보이는가?
- flow arrow가 실제 벌 이동 방향과 맞는가?
- 노이즈 구간에서 filtered candidate가 과도하게 남지 않는가?
- 활동 구간에서 실제 벌 움직임이 filter 때문에 사라지지 않는가?

CSV에서 확인할 것:

- raw flux보다 filtered flux가 적절히 줄었는가?
- 조용한 영상의 filtered traffic이 낮고 안정적인가?
- 활발한 영상의 filtered traffic은 충분히 살아 있는가?
- `filtered_in_count_est`, `filtered_out_count_est`가 truth label과 비례하는가?

---

## 24. 현재 가장 중요한 실험 질문

1. 현재 ROI와 entrance 좌표가 모든 영상에 맞는가?
2. 코드의 IN/OUT 방향 정의가 실제 벌의 입장/퇴장과 일치하는가?
3. 조용한 영상에서 filtered flux가 충분히 낮은가?
4. 활동이 많은 영상에서 실제 벌 움직임까지 지나치게 제거하지 않는가?
5. `in_bee_flux_unit`, `out_bee_flux_unit`을 어떤 truth label로 calibration할 것인가?
6. `selected` preset이 실제 운영 기준으로 적절한가?

---

## 25. 앞으로의 작업 제안

### 1단계: 좌표 검증

- 각 영상에서 ROI와 entrance rectangle이 맞는지 preview로 확인한다.
- `roi_ent_info.txt`의 좌표와 `Config` 기본값 차이를 정리한다.

### 2단계: 조용한 영상 기준 노이즈 억제

- low-activity 영상에서 raw/filtered flux 차이를 비교한다.
- `blur_kernel`, `flow_mag_threshold`, `normal_flow_threshold`, `persist_threshold`, `min_flow_component_area`를 순서대로 조정한다.

### 3단계: 활동 영상 기준 과소탐지 방지

- 실제 벌 출입이 잘 보이는 영상에서 filtered candidate가 살아 있는지 확인한다.
- 너무 강한 area filter나 threshold가 실제 움직임을 지우지 않는지 본다.

### 4단계: Truth CSV 생성

아래 형식으로 사람이 직접 센 정답 파일을 만든다.

```csv
video,in,out
ANU-25-summer-6_20260328_130000.mp4,12,8
ANU-25-summer-11_20260328_130000.mp4,3,5
```

### 5단계: 평가와 튜닝

```bash
uv run python -m src.main --mode tune --preset selected --truth-csv videos/entrance.csv
```

평가 결과를 보고 active 영상의 오차와 zero/quiet 영상의 false positive를 함께 줄인다.

---

## 26. 이 프로젝트를 설명할 때의 핵심 메시지

이 프로젝트는 "벌을 한 마리씩 찾는 문제"가 아니라 "입구 경계를 통과하는 움직임의 흐름을 안정적으로 측정하는 문제"다.

따라서 가장 중요한 것은 세 가지다.

1. 입구 경계를 정확히 잡는 것
2. 경계를 통과하는 방향성 있는 움직임만 flux로 합산하는 것
3. 노이즈는 줄이되 실제 벌 움직임은 보존하는 필터를 찾는 것

현재 코드는 이 세 가지를 실험하고 비교하기 위한 기반을 갖춘 상태다.
