# Bee Entrance Optical Flow

벌통 입구 영상에서 벌의 출입 활동량을 추정하기 위한 OpenCV optical-flow 실험 디렉토리입니다. 현재 구현의 중심은 개별 벌을 탐지하거나 ID tracking을 하는 것이 아니라, 입구 경계 주변의 optical flow를 이용해 `IN`, `OUT` 방향 flux를 계산하고 3초 단위 활동량으로 요약하는 것입니다.

## 현재 진행 요약

- Farneback optical flow 기반으로 ROI 내부 움직임을 계산했습니다.
- 벌통 입구 사각형의 top, bottom, left, right 네 경계를 모두 counting boundary로 사용하도록 구성했습니다.
- 입구 하단이 ROI 하단과 맞닿아 있더라도 bottom edge 주변의 optical flow를 함께 누적해 4방향 출입 flux를 계산합니다.
- 프레임 단위 raw flux를 그대로 누적하면 노이즈와 느린 움직임이 과대 계수되는 문제가 있어, 먼저 flux 신호를 안정화하는 방향으로 전환했습니다.
- Gaussian blur, temporal persistence filter, optional connected-component area filter를 추가해 짧은 optical-flow 스파이크와 작은 노이즈 성분을 줄였습니다.
- `src/main.py`에 batch, compare, groups, evaluate, tune 실행 모드를 추가해 여러 영상과 파라미터 preset을 비교할 수 있게 했습니다.
- 실험 결과 확인을 위해 preview video, frame CSV, window CSV, comparison/evaluation CSV를 생성하도록 정리했습니다.

## 디렉토리 구조

```text
.
├── src/
│   ├── bee_entrance_count.py
│   ├── main.py
│   ├── optical_count.py
│   ├── capture.py
│   ├── vis.py
│   ├── vis_arr.py
│   ├── vis_arr2.py
│   └── sep_blob.py
├── docs/
├── videos/
├── bee_count_output/
├── pyproject.toml
└── README.md
```

## 주요 파일

| 파일 | 역할 |
| --- | --- |
| `src/bee_entrance_count.py` | optical-flow 계산, 경계 flux 계산, persistence/component filter, CSV 및 preview 출력의 핵심 구현 |
| `src/main.py` | 여러 영상을 batch/compare/groups/evaluate/tune 모드로 실행하는 편의 runner |
| `src/optical_count.py` | 기본 비교 영상 2개를 실행하는 작은 wrapper |
| `src/capture.py` | preview 영상에서 특정 프레임을 이미지로 추출해 ROI/overlay를 확인하는 보조 스크립트 |
| `src/vis.py`, `src/vis_arr.py`, `src/vis_arr2.py` | optical-flow magnitude, arrow, HSV 방향 시각화 실험 스크립트 |
| `src/sep_blob.py` | flow magnitude 기반 blob/component 분리 실험 스크립트 |
| `docs/` | noise filtering, persistence filter, event counting 접근에 대한 상세 메모 |

## 처리 흐름

1. 입력 영상을 OpenCV로 읽습니다.
2. 고정 ROI를 crop합니다.
3. 전체 프레임 기준 entrance rectangle을 ROI 좌표로 변환합니다.
4. entrance rectangle의 네 변을 기준으로 boundary band와 inward normal vector를 계산합니다.
5. 입구 top/bottom/left/right 주변 boundary band를 counting 영역으로 사용합니다.
6. 연속 프레임 사이의 Farneback optical flow를 계산합니다.
7. flow를 boundary normal 방향으로 투영해 `IN`/`OUT` flux를 분리합니다.
8. raw candidate mask에 persistence filter와 선택적 component area filter를 적용합니다.
9. raw/filtered flux를 프레임 CSV로 저장하고, 3초 window 단위 count estimate를 생성합니다.
10. ROI preview video에 entrance rectangle, counting band, flow 후보를 시각화하고, flux 값은 별도 정보 패널에 표시합니다.

## 기본 좌표와 파라미터

현재 기본 ROI와 입구 좌표는 `Config`에 고정되어 있습니다.

```python
roi_x1, roi_y1, roi_x2, roi_y2 = 1300, 1000, 1640, 1232
ent_x1, ent_y1, ent_x2, ent_y2 = 1400, 1200, 1600, 1232
```

주요 기본값은 다음과 같습니다.

| 파라미터 | 기본값 | 의미 |
| --- | ---: | --- |
| `boundary_band_px` | `8` | 입구 경계 주변 counting band 폭 |
| `flow_mag_threshold` | `0.30` | 후보 pixel로 인정할 최소 flow magnitude |
| `normal_flow_threshold` | `0.08` | 경계 normal 방향 최소 flow |
| `preview_panel_width` | `360` | preview 영상 오른쪽 정보 패널 폭 |
| `blur_kernel` | `3` | grayscale frame Gaussian blur kernel |
| `use_persistence_filter` | `True` | 짧은 one-frame noise 억제 |
| `persist_decay` | `0.65` | persistence map 감쇠율 |
| `persist_threshold` | `1.3` | 후보가 통과하기 위한 persistence threshold |
| `use_component_area_filter` | `False` | component 면적 필터 사용 여부 |
| `min_flow_component_area` | `30` | component area filter 최소 면적 |
| `window_sec` | `3.0` | 요약 CSV window 길이 |

`src/main.py`의 현재 preset은 `raw`, `persistence`, `blur5`, `strict_noise`, `selected`입니다. 기본 preset은 `selected`이며 blur 5, threshold 강화, area filter를 함께 사용합니다.

## 설치

Python 3.11 이상을 사용합니다. `pyproject.toml` 기준 의존성은 다음과 같습니다.

- `opencv-python`
- `numpy`
- `pandas`

`uv`를 사용하는 경우:

```powershell
uv sync
```

일반 `pip` 환경에서는:

```powershell
pip install opencv-python numpy pandas
```

## 실행 예시

단일 영상 처리:

```powershell
python -m src.bee_entrance_count --video videos/ANU-25-summer-6_20260405_060000.mp4
```

두 개 이상의 영상 비교:

```powershell
python -m src.bee_entrance_count --compare videos/ANU-25-summer-6_20260405_060000.mp4 videos/ANU-25-summer-6_20260405_070000.mp4
```

현재 선택 preset으로 전체 batch 실행:

```powershell
python -m src.main --mode batch --preset selected
```

특정 영상 목록만 비교:

```powershell
python -m src.main --mode compare --videos videos/ANU-25-summer-6_20260405_060000.mp4 videos/ANU-25-summer-6_20260405_070000.mp4
```

간단한 parameter grid tuning:

```powershell
python -m src.main --mode tune --preset selected --truth-csv videos/entrance.csv
```

실행 계획만 확인:

```powershell
python -m src.main --mode batch --preset selected --dry-run
```

## 산출물

기본 산출물은 `bee_count_output/` 아래에 생성됩니다. 이 디렉토리는 `.gitignore`에 포함되어 있어 Git에는 올라가지 않습니다.

단일 영상 처리 시 주요 파일:

- `{video_stem}_preview.mp4`: ROI와 별도 정보 패널을 함께 담은 preview 영상
- `{video_stem}_frame_flux.csv`: 프레임별 raw/filtered flux
- `{video_stem}_window_3sec.csv`: 3초 window별 count estimate

비교 및 batch 실행 시 주요 파일:

- `comparison_summary.csv`: 여러 영상의 raw/filtered traffic flux 비교
- `batch_summary.csv`: batch 처리 결과 요약
- `group_summary.csv`: sliding group 비교 결과
- `evaluation.csv`, `evaluation_metrics.csv`: truth CSV와 비교한 평가 결과
- `tuning_results.csv`: parameter grid별 평가 결과

## 문서화된 실험 기록

상세한 판단과 변경 배경은 `docs/`에 나누어 기록했습니다.

- `docs/bee_entrance_event_counting.md`: event detector 기반 counting 접근과 한계
- `docs/bee_entrance_noise_filtering.md`: 노이즈 우선 제거 전략과 component filter 실험
- `docs/bee_entrance_persistence_filter.md`: 현재 persistence 중심 필터 구조와 tuning 순서

현재 방향은 event detector를 잠시 되돌리고, 조용한 영상에서 filtered flux가 작고 안정적으로 유지되도록 optical-flow 신호 자체를 먼저 정리하는 것입니다. 이후 실제 벌 이동이 뚜렷한 positive sample에서 과도하게 억제되지 않는지 확인해야 합니다.

## 데이터 관리

`videos/`와 `bee_count_output/`은 로컬 데이터/생성물 디렉토리로 관리합니다. 영상 원본, preview, CSV 결과물은 크기가 커질 수 있으므로 기본적으로 Git 추적에서 제외되어 있습니다.

## 시각화 보조 스크립트

`src/vis.py`는 ROI의 optical-flow magnitude를 heatmap으로 겹쳐 저장합니다. `src/vis_arr.py`는 일정 간격의 flow vector를 화살표로 표시해 움직임 방향을 빠르게 확인하는 용도입니다. `src/vis_arr2.py`는 flow 방향을 HSV hue로, 강도를 value로 표현해 전체 flow field의 방향 분포를 확인합니다. `src/sep_blob.py`는 flow magnitude mask에 morphology와 connected component 분석을 적용해 움직임 blob 후보가 어떻게 분리되는지 보는 실험용 스크립트입니다.
