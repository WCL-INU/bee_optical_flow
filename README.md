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
| `src/extract_video_features.py` | 참값 없이 영상에서 밝기/선명도/flow/component/방향 혼잡도 feature를 추출하는 batch 도구 |
| `src/optical_count.py` | 기본 비교 영상 2개를 실행하는 작은 wrapper |
| `src/capture.py` | preview 영상에서 특정 프레임을 이미지로 추출해 ROI/overlay를 확인하는 보조 스크립트 |
| `src/vis.py`, `src/vis_arr.py`, `src/vis_arr2.py` | optical-flow magnitude, arrow, HSV 방향 시각화 실험 스크립트 |
| `src/sep_blob.py` | flow magnitude 기반 blob/component 분리 실험 스크립트 |
| `docs/` | noise filtering, persistence filter, event counting 접근에 대한 상세 메모 |

## 처리 흐름

1. 입력 영상을 OpenCV로 읽습니다.
2. 설정된 ROI를 crop합니다.
3. 전체 프레임 기준 entrance rectangle을 ROI 좌표로 변환합니다.
4. entrance rectangle의 네 변을 기준으로 boundary band와 inward normal vector를 계산합니다.
5. 입구 top/bottom/left/right 주변 boundary band를 counting 영역으로 사용합니다.
6. 연속 프레임 사이의 Farneback optical flow를 계산합니다.
7. flow를 boundary normal 방향으로 투영해 `IN`/`OUT` flux를 분리합니다.
8. raw candidate mask에 persistence filter와 선택적 component area filter를 적용합니다.
9. raw/filtered flux를 프레임 CSV로 저장하고, 3초 window 단위 count estimate를 생성합니다.
10. ROI preview video에 entrance rectangle, counting band, flow 후보를 시각화하고, flux 값은 별도 정보 패널에 표시합니다.

## 좌표와 파라미터

ROI와 입구 경계 좌표는 전체 프레임 기준 `(x1, y1, x2, y2)` 사각형입니다. `Config`의 기본 좌표는 `ANU-25-summer-20` 기준이며, `src/main.py`에서는 영상명에 맞는 좌표 preset을 자동 적용하거나 CLI에서 직접 덮어쓸 수 있습니다.

```python
roi_x1, roi_y1, roi_x2, roi_y2 = 1020, 980, 1420, 1280
ent_x1, ent_y1, ent_x2, ent_y2 = 1120, 1080, 1320, 1180
```

`src/main.py`의 `--coordinate-preset auto`가 기본값입니다. 현재 `roi_ent_info.txt`에 정리된 `anu25_summer_3`, `anu25_summer_5`, `anu25_summer_7`, `anu25_summer_9`, `anu25_summer_12`, `anu25_summer_13`, `anu25_summer_14`, `anu25_summer_15`, `anu25_summer_16`, `anu25_summer_20` 좌표를 영상 파일명 기준으로 매칭합니다. 매칭되는 preset이 없으면 `Config` 기본 좌표를 사용합니다.

좌표 관련 실행 옵션:

| 옵션 | 의미 |
| --- | --- |
| `--coordinate-preset auto` | 영상명으로 좌표 preset 자동 선택 |
| `--coordinate-preset anu25_summer_15` | 특정 좌표 preset 강제 사용 |
| `--coordinate-preset default` | preset 적용 없이 `Config` 기본 좌표 사용 |
| `--roi X1 Y1 X2 Y2` | ROI 좌표 직접 지정 |
| `--entrance X1 Y1 X2 Y2` | 입구 경계 좌표 직접 지정 |
| `--boundary-band-px N` | 입구 경계 주변 counting band 폭 지정 |

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
uv run python -m src.bee_entrance_count --video videos/ANU-25-summer-6_20260405_060000.mp4
```

두 개 이상의 영상 비교:

```powershell
uv run python -m src.bee_entrance_count --compare videos/ANU-25-summer-6_20260405_060000.mp4 videos/ANU-25-summer-6_20260405_070000.mp4
```

현재 선택 preset으로 전체 batch 실행:

```powershell
uv run python -m src.main --mode batch --preset selected
```

영상명에 맞는 좌표 preset을 자동 적용해서 batch 실행:

```powershell
uv run python -m src.main --mode batch --video-dir videos --pattern "ANU-25-summer-*.mp4" --coordinate-preset auto
```

특정 기기/영상군 좌표를 강제로 사용:

```powershell
uv run python -m src.main --mode batch --coordinate-preset anu25_summer_15 --video-dir videos --pattern "ANU-25-summer-15_*.mp4"
```

ROI와 입구 경계 좌표를 직접 지정:

```powershell
uv run python -m src.main --mode batch --videos videos/ANU-25-summer-20_20260328_130000.mp4 --roi 1020 980 1420 1280 --entrance 1120 1080 1320 1180 --boundary-band-px 8
```

특정 영상 목록만 비교:

```powershell
uv run python -m src.main --mode compare --videos videos/ANU-25-summer-6_20260405_060000.mp4 videos/ANU-25-summer-6_20260405_070000.mp4
```

간단한 parameter grid tuning:

```powershell
uv run python -m src.main --mode tune --preset selected --truth-csv videos/entrance.csv
```

실행 계획만 확인:

```powershell
uv run python -m src.main --mode batch --preset selected --dry-run
```

## 산출물

기본 산출물은 `bee_count_output/` 아래에 생성됩니다. 이 디렉토리는 `.gitignore`에 포함되어 있어 Git에는 올라가지 않습니다.

단일 영상 처리 시 주요 파일:

- `{video_stem}_preview.mp4`: ROI와 별도 정보 패널을 함께 담은 preview 영상
- `{video_stem}_frame_flux.csv`: 프레임별 raw/filtered flux
- `{video_stem}_window_3sec.csv`: 3초 window별 count estimate

`batch_summary.csv`와 `comparison_summary.csv`에는 처리에 사용된 `roi_*`, `ent_*`, `boundary_band_px` 컬럼도 함께 저장됩니다.

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

## 영상 feature 추출

`src/extract_video_features.py`는 실제 이출입량 같은 참값을 사용하지 않고, 영상 자체에서 측정 가능한 신뢰도/오차 원인 후보 feature를 추출하는 배치 스크립트입니다. 서버에 원본 영상이 있을 때 이 스크립트를 실행하고, 생성된 CSV를 로컬로 가져와 회귀 오차와의 상관관계를 분석하는 용도로 사용합니다.

추출하는 주요 feature 범위:

- ROI, 입구 영역, 배경 영역의 밝기 평균/분산, 분위수, dynamic range, 어두운 픽셀 비율, 밝은 픽셀 비율, 포화 픽셀 비율, entropy
- HSV saturation/value 통계
- blur/선명도 지표: Laplacian variance, Tenengrad, edge density
- 연속 프레임 차이: frame difference 평균, p90, 변화 픽셀 비율
- optical-flow magnitude, normal-flow, active pixel 비율, 방향 entropy
- raw/persistent/filtered candidate pixel 수와 component 수/면적 통계
- top/bottom/left/right 경계별 raw/filtered in/out flux
- filtered/raw retention, in/out share, direction balance 같은 혼잡도/방향 분리 후보 지표

서버에서 전체 영상 feature를 추출하는 기본 실행 예:

```powershell
uv run python -m src.extract_video_features ^
  --video-dir D:\bee_videos ^
  --pattern "ANU-25-summer-*.mp4" ^
  --output-dir analysis\video_features\output ^
  --preset selected ^
  --coordinate-preset auto
```

여러 패턴을 한 번에 지정할 수도 있습니다. 같은 파일이 여러 패턴에 중복 매칭되면 한 번만 처리됩니다.

```powershell
uv run python -m src.extract_video_features ^
  --video-dir D:\bee_videos ^
  --pattern "ANU-25-summer-3_*.mp4" "ANU-25-summer-12_*.mp4" "ANU-25-summer-20_*.mp4" ^
  --output-dir analysis\video_features\output ^
  --preset selected ^
  --coordinate-preset auto
```

빠른 시험 실행이 필요하면 일부 프레임만 샘플링할 수 있습니다. `--frame-stride 5`는 5 frame pair마다 한 번만 feature를 계산하므로 처리 시간이 줄어듭니다.

```powershell
uv run python -m src.extract_video_features ^
  --video-dir D:\bee_videos ^
  --pattern "ANU-25-summer-*.mp4" ^
  --output-dir analysis\video_features\output_stride5 ^
  --preset selected ^
  --coordinate-preset auto ^
  --frame-stride 5
```

특정 영상만 처리:

```powershell
uv run python -m src.extract_video_features ^
  --videos videos\ANU-25-summer-3_20260318_150000.mp4 ^
  --output-dir analysis\video_features\output_one ^
  --preset selected
```

실행 계획만 확인:

```powershell
uv run python -m src.extract_video_features ^
  --video-dir D:\bee_videos ^
  --pattern "ANU-25-summer-*.mp4" ^
  --preset selected ^
  --dry-run
```

프레임별 상세 CSV까지 저장하려면 `--write-frame-csv`를 추가합니다. 전체 영상에 대해 프레임별 CSV를 저장하면 파일이 매우 커질 수 있으므로, 보통은 비디오별 요약 CSV와 window CSV만 먼저 생성합니다.

주요 옵션:

| 옵션 | 설명 |
| --- | --- |
| `--video-dir DIR` | 영상 파일이 있는 디렉토리 |
| `--pattern PATTERN ...` | 처리할 영상 glob 패턴. 여러 개 지정 가능 |
| `--videos PATH ...` | 특정 영상 목록 직접 지정 |
| `--output-dir DIR` | feature CSV 산출 디렉토리 |
| `--preset selected` | `src/main.py`의 처리 preset 재사용 |
| `--coordinate-preset auto` | 영상 파일명으로 ROI/입구 좌표 preset 자동 선택 |
| `--roi X1 Y1 X2 Y2` | ROI 좌표 직접 지정 |
| `--entrance X1 Y1 X2 Y2` | 입구 경계 좌표 직접 지정 |
| `--frame-stride N` | N frame pair마다 한 번 feature 계산 |
| `--max-frame-pairs N` | 테스트용 최대 frame pair 수 제한 |
| `--window-sec SEC` | window 요약 길이. 기본값은 120초 |
| `--write-frame-csv` | 프레임별 feature CSV 저장 |

생성 결과:

- `video_image_flow_features.csv`: 비디오별 요약 feature. 기존 검증 데이터와 `video` 컬럼으로 join할 수 있습니다.
- `video_image_flow_features_windows.csv`: 비디오/window별 feature. 시간 구간별 오차 원인을 볼 때 사용합니다.
- `video_image_flow_features_dictionary.csv`: feature 컬럼군 설명.
- `{video_stem}_image_flow_features_frame.csv`: `--write-frame-csv` 사용 시 생성되는 프레임별 상세 feature.

## Feature/Error Analysis Utility

`src/analyze_video_feature_errors.py`는 비디오 feature, 검증 카운트, 회귀 모델 계수를 합쳐 분석용 테이블을 만들고, 원하는 feature/target 쌍의 상관관계를 계산합니다.

기기 3번의 선형회귀 오차와 feature 상관관계 분석:

```powershell
uv run python -m src.analyze_video_feature_errors --device 3 --top 20 --scatter
```

주요 옵션:

| 옵션 | 설명 |
| --- | --- |
| `--features PATH` | 입력 feature CSV. 기본값은 `analysis/video_features/output/video_image_flow_features.csv` |
| `--truth PATH` | 검증 카운트/flux 테이블. CSV/XLSX 지원 |
| `--models PATH` | 회귀 모델 비교 CSV. 기본값은 `validation/output/regression_model_comparison.csv` |
| `--device 3` | 특정 기기만 분석. 여러 기기는 `--device 3 5` 또는 `--device 3,5` |
| `--targets COL ...` | 분석할 오차/target 컬럼. 기본값은 `total_abs_error`, `sum_abs_error`, `in_abs_error`, `out_abs_error` |
| `--feature-regex REGEX` | 특정 feature 이름 패턴만 분석 |
| `--method pearson|spearman` | 상위 결과 정렬 기준 상관계수 |
| `--outlier-filter none|iqr|zscore` | target-feature 상관계수 계산 전 쌍별 이상치 제거 |
| `--outlier-iqr-multiplier 1.5` | IQR 이상치 제거 기준 배수 |
| `--outlier-z-threshold 3.0` | z-score 이상치 제거 기준 |
| `--feature-correlations` | feature-feature 쌍별 Pearson/Spearman 상관계수도 함께 계산 |
| `--feature-correlation-max-abs 0.999` | 출력 요약에서 완전중복에 가까운 feature 쌍을 제외하고 볼 기준 |
| `--group-features` | 높은 상관 feature들을 그룹화하고 그룹별 대표 feature 선택 |
| `--feature-group-threshold 0.95` | feature 그룹화에 사용할 절대 상관계수 기준 |
| `--representative-scatter` | 대표 feature와 오차 target의 상위 쌍 산포도 PNG 저장 |
| `--combine-representatives` | 대표 feature들을 조합한 ridge 선형 score를 target별로 교차검증 평가 |
| `--combo-top-features 20` | 조합 모델에 사용할 target별 상위 대표 feature 수 |
| `--combo-alpha 10` | 조합 모델 ridge 정규화 강도 |
| `--combo-folds 5` | 조합 모델 교차검증 fold 수 |
| `--covariance` | target-feature 쌍별 공분산도 함께 계산 |
| `--covariance-normalization none|zscore|minmax|robust` | 공분산 계산 전 feature/target 정규화 방식. `zscore` 공분산은 Pearson 상관계수와 같습니다 |
| `--scatter` | 상위 feature-target 쌍 산점도 PNG 저장 |

생성 결과:

- `{device}_video_features_with_linear_errors.csv`: feature, 실제 카운트, 예측값, signed/absolute error가 결합된 분석용 테이블.
- `{device}_feature_error_correlations.csv`: target-feature 쌍별 Pearson/Spearman 상관계수.
- `{device}_feature_feature_correlations.csv`: `--feature-correlations` 사용 시 생성되는 feature-feature 쌍별 Pearson/Spearman 상관계수.
- `{device}_feature_correlation_groups.csv`: `--group-features` 사용 시 생성되는 유사 feature 그룹 요약.
- `{device}_representative_features.csv`: `--group-features` 사용 시 생성되는 그룹별 대표 feature.
- `{device}_representative_feature_error_scatter.png`: `--representative-scatter` 사용 시 생성되는 대표 feature/error 산포도.
- `{device}_representative_feature_error_models.csv`: `--combine-representatives` 사용 시 생성되는 target별 조합 모델 평가.
- `{device}_representative_feature_error_predictions.csv`: `--combine-representatives` 사용 시 생성되는 조합 모델 교차검증 예측값.
- `{device}_representative_feature_error_model_scatter.png`: `--combine-representatives` 사용 시 생성되는 조합 score/error 산포도.
- `{device}_feature_error_covariances.csv`: `--covariance` 사용 시 생성되는 target-feature 쌍별 공분산.
- `{device}_feature_error_scatter.png`: `--scatter` 사용 시 생성되는 상위 쌍 산점도.

Python 코드에서 직접 사용할 때는 `build_feature_error_table`, `numeric_columns`, `pairwise_correlations`, `pairwise_covariances`, `top_correlations`, `top_covariances`, `merge_extra_tables`를 import해서 여러 테이블과 컬럼쌍을 조합할 수 있습니다.

## Validation Viewer

검증 데이터 뷰어는 `validation/build_data_viewer.py`가 담당합니다. `validation/data/merged_data.xlsx`를 읽고, 선형 회귀와 flat-exponential 회귀를 계산한 뒤 정적 HTML 뷰어를 생성합니다.

```powershell
uv run python validation\build_data_viewer.py
```

생성 결과:

- `validation/output/data_viewer.html`
- `validation/output/regression_model_comparison.csv`

검증 폴더 구조:

- `validation/build_data_viewer.py`: 현재 사용하는 validation viewer 빌더
- `validation/data/`: 입력 spreadsheet와 검증 원천 데이터
- `validation/output/`: 재생성 가능한 HTML/CSV 산출물
- `validation/legacy/`: 이전 정적 그래프/회귀 리포트 보관

자세한 사용법은 `validation/data_viewer.md`를 참고합니다.
