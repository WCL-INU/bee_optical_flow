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

여러 영상을 프로세스 4개로 병렬 처리:

```powershell
uv run python -m src.main --mode batch --preset selected --workers 4
```

`--workers`의 기본값은 `1`이며, `tune` 모드에서도 각 trial 안의 영상들을 같은 방식으로 병렬 처리합니다. CPU 코어 수와 디스크 읽기 속도를 고려해 값을 조절하세요.

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

## 기기별 화면 구도 변경 감시

`src/detect_scene_changes.py`는 같은 기기의 영상을 촬영 시각 순서로 비교하여
**어느 영상부터 화면 구도가 바뀌었는지** 보고합니다. 영상 내부의 변경 시점을
찾거나 ROI를 수정하지 않습니다. 파일명은 `기기명_YYYYMMDD_HHMMSS.mp4` 형식입니다.

```bash
# 특정 기기의 영상 전체 비교
uv run python -m src.detect_scene_changes --device ANU-25-summer-3

# 처음 6개 영상으로 시험 (limit은 기기별 적용)
uv run python -m src.detect_scene_changes --device ANU-25-summer-3 --limit 6

# 촬영 날짜 범위 지정 (시작일과 종료일 포함)
uv run python -m src.detect_scene_changes --device ANU-25-summer-3 --start-date 2026-03-01 --end-date 2026-03-31

# 일시 범위 지정 (양쪽 시각 포함)
uv run python -m src.detect_scene_changes --device ANU-25-summer-3 --start-datetime "2026-03-01 09:30:00" --end-datetime "2026-03-05 18:00:00"

# 디렉토리 안의 모든 기기를 각각 비교
uv run python -m src.detect_scene_changes --video-dir videos
```

영상당 기본 21개 프레임을 균등 추출하여 시간 중앙값 배경을 만들고, 밝기 변화를
보정한 뒤 변동이 큰 픽셀 주변을 비교에서 제외합니다. 움직이는 벌과 흔들리는
풀의 영향을 줄이기 위한 처리이며, 계속 같은 위치를 가리는 벌 무리는 남을 수 있습니다.

날짜는 파일명의 촬영 시각을 기준으로 하며 종료일은 하루 전체를 포함합니다.
시작일/종료일 중 하나만 지정할 수도 있습니다. 날짜 필터 이후 기기별 `--limit`이
적용되며, 선택 기간 안의 첫 비교 가능한 영상이 기준 화면이 됩니다. 따라서 기간
시작 전에 발생한 변화는 이 실행에서 판정하지 않습니다.

시간까지 지정하려면 `--start-datetime` / `--end-datetime`을 사용합니다.
`YYYY-MM-DD HH:MM:SS`, `YYYY-MM-DD HH:MM` 및 공백 대신 `T`를 쓰는 형식을
지원하며 초를 생략하면 `00`초입니다. 파일명 시각 그대로 비교하며 시간대 변환은
하지 않습니다. 양쪽 시각을 포함하고, 한쪽만 지정하거나 시작 날짜와 종료 일시를
조합할 수도 있습니다. 같은 쪽의 날짜 옵션과 일시 옵션은 함께 사용할 수 없습니다.

먼저 Canny/HoughLinesP로 긴 선분을 추출하고 위치·방향·겹치는 길이로 일대일
대응합니다. 짧은 선과 변동 영역에 걸친 선을 제외하고 두꺼운 경계의 중복 선을
합칩니다. 최소 3개의 대응 선과 양쪽 선분 수 대비 65% 이상 일치, 서로 다른 방향의
경계가 확인되면 동일 구도로 판단합니다. 평행선만으로는 선 방향으로의 이동을
판단할 수 없으므로 이 조건을 충족하지 않습니다. 이후 고정 경계 일치, ECC 정합,
SIFT 특징점과 RANSAC으로 배경의 대응 관계를 확인합니다.

특징점 대응 소실만으로는 더 이상 구도 변경으로 판정하지 않습니다. 젖은 땅이나
그림자처럼 표면만 달라질 수 있기 때문입니다. 대응점을 잃었을 때는 양쪽에 충분한
서로 다른 방향의 긴 선들이 있고 선 배치도 거의 대응하지 않는 경우에만 변경 후보로
삼으며, 나머지는 `unknown`으로 보류합니다. 선을 벌통 경계로 의미적으로 분류하는
모델은 아니므로 풀·그림자가 만든 선이나 가려진 경계로 인한 오판 가능성은 남습니다.

첫 비교 가능한 영상을 기준으로 유지하다가, 변경 후보와 같은 새 구도가 후속
영상에서도 확인되면 최초 후보 영상부터 변경으로 보고하고 새 기준을 설정합니다.
기본 확인 수는 2개 영상이며 `--confirmations`로 늘릴 수 있습니다. 어둡거나 배경
특징이 부족한 영상은 `unknown`으로 기록하고 확인 수에 포함하지 않습니다.
한 번 달라졌다가 기존 구도로 돌아온 경우는 변경 보고에서 제외합니다.

산출물은 기본 `bee_count_output/scene_changes/`에 저장됩니다.

- `report.md`: 기기별 최초 변경 영상, 마지막 정상 영상, 확인 영상과 전후 이미지 링크
- `scene_changes.json`: 변경 이벤트, 후속 근거 부족으로 확인 대기 중인 후보, 실행 설정
- `video_checks.csv`: 모든 영상의 판정, 판단 불가 이유, `reference_lines`, `current_lines`, `matched_lines`, `line_match_ratio`
- `*_change.jpg`: 변경 전후 대표 배경. 빨간 부분은 제외한 변동 영역, 청록색은 추출한 긴 선분(모두 대응에 성공했다는 뜻은 아님)

`--samples`는 영상당 샘플 수, `--max-width`는 분석 이미지 최대 너비(기본 800),
`--shift-fraction`은 이미지 대각선 대비 이동 허용치(기본 0.015)입니다.
수치는 초기 휴리스틱이며 실제 변경 이력으로 검증해야 합니다. 대응점과 선 배치가
함께 달라지는 큰 장면 변화도 후보가 되지만, 심한 조명 변화·가림·계절 변화와 완전히 구분된다는
보장은 없습니다. 보고하는 시간은 파일명에 기록된 영상 시각이며, 촬영 공백 중
실제 변경 시점은 마지막 정상 영상과 최초 변경 영상 사이에 있습니다.

검증 실행: `uv run python -m unittest discover -s tests -v`

### 저장된 타임랩스로 구도 변경 감지

`device1.mp4`~`device20.mp4`와 같은 이름의 CSV가 있는 경우 원본 영상을 다시 열지
않고 분석할 수 있습니다. CSV의 `output_start_sec`와 `output_frames`로 원본 영상별
구간을 나누고, `video_start`의 촬영 일시로 변경 시점을 보고합니다. 예전 CSV처럼
`backend`, `sampling` 열이 없는 파일도 지원합니다.

```bash
# 기본적으로 device1~device20 전체 선택
uv run python -m src.detect_scene_changes --timelapse-dir bee_count_output/timelapse --workers 2

# 특정 파일과 원본 촬영 일시 범위만 선택
uv run python -m src.detect_scene_changes --timelapse-dir bee_count_output/timelapse --timelapse-ids 1 3 20 --start-date 2026-07-05 --end-date 2026-07-10 --workers 2

# 각 파일의 첫 2개 원본 구간만으로 시험
uv run python -m src.detect_scene_changes --timelapse-dir bee_count_output/timelapse --limit 2 --output-dir bee_count_output/timelapse_scene_changes_smoke
```

기본 출력은 `bee_count_output/timelapse_scene_changes/`입니다. 최상위 `report.md`와
`scene_changes.json`은 전체 요약이고, 각 `deviceN/` 아래에 개별 보고서,
`video_checks.csv`, 변경 전후 이미지가 저장됩니다. 변경 이벤트에는 원본 촬영 일시와
타임랩스 재생 위치(초)가 함께 기록됩니다. 원본 검사 모드와 같은 날짜·일시 범위,
샘플 수, 이동 임계값, 후속 확인 개수 옵션을 사용할 수 있습니다.

`--workers`는 이 모드에서 동시에 처리할 **기기 파일 수**입니다. 각 파일은 한 번
열어 원본 구간 순서대로 읽으며, 기존 원본 영상 배경 캐시는 사용하지 않습니다.
`--limit`은 날짜 필터를 적용한 뒤 기기별 원본 구간 수를 제한합니다.

날짜 캡션 자체가 구도 특징으로 인식되지 않도록 화면 상단 12%를 잘라낸 뒤 비교합니다.
캡션 위치가 다르면 `--caption-fraction`을 조정하세요. CSV/MP4의 프레임 개수와
구간 연속성을 확인하며, 매핑이 맞지 않으면 추측하지 않고 해당 기기를 오류로 보고합니다.
`error.json`을 남기고 나머지 기기를 처리한 뒤 종료 코드 1을 반환합니다.

`--sampling first`로 만든 타임랩스에는 원본 앞부분만 들어 있습니다. 따라서 원본
2분에서 얻는 시간적 배경보다 벌·가림을 배제할 근거가 적고, 압축과 상단 잘라내기도
판정에 영향을 줄 수 있습니다. 타임랩스만 분석한 결과를 원본 전체 분석과 동일한
정확도로 해석해서는 안 됩니다.

### 구도 감지 실행 속도

영상마다 21개 위치로 이동해서 압축 프레임을 읽는 작업이 주된 비용입니다.
대표 배경·마스크·SIFT 특징을 기본 `bee_count_output/scene_background_cache/`에
캐시하므로, 같은 영상을 다시 검사할 때 디코딩과 배경 생성을 건너뜁니다.
영상의 경로·크기·수정 시각, 샘플 수, 분석 해상도, 캐시 버전 및 라이브러리 버전이
같을 때만 재사용합니다. 판정 임계값이나 검사 기간만 바꾸는 경우에는 재사용됩니다.
캐시는 디스크 공간을 사용하며 해당 디렉토리를 지워도 다음 실행에서 재생성됩니다.
전처리 알고리즘을 수정하는 개발자는 `BACKGROUND_CACHE_VERSION`도 올려야 합니다.

```bash
# 처음 처리하는 영상은 2개씩 준비. 판정은 여전히 촬영 순서대로 실행
uv run python -m src.detect_scene_changes --device ANU-25-summer-3 --workers 2

# 캐시 위치 지정 또는 캐시를 끄고 검증
uv run python -m src.detect_scene_changes --device ANU-25-summer-3 --cache-dir bee_count_output/my_cache
uv run python -m src.detect_scene_changes --device ANU-25-summer-3 --no-cache
```

`--workers` 기본값은 1입니다. 병렬 준비 개수는 제한되어 있어 전체 영상의 배경을
메모리에 한꺼번에 적재하지 않습니다. CPU·디스크 상황에 따라 병렬 실행이 더 느릴
수 있으므로 2부터 비교하세요. `video_checks.csv`의 `cache_hit`, `background_sec`,
`comparison_sec`로 캐시 사용 여부와 영상별 처리 시간을 확인할 수 있습니다.
병렬 실행의 영상별 시간 합계는 전체 실행 시간과 다릅니다.
기본 샘플 수와 분석 해상도는 그대로 유지합니다. 이를 줄이면 빨라질 수 있지만
벌을 배제하는 효과와 작은 구도 변화의 감지 성능이 달라질 수 있습니다.

## 기기별 타임랩스 만들기

한 기기의 영상을 촬영 일시 순서대로 연결하고, 기본적으로 각 영상의 **앞 24프레임을
연속으로 추출**합니다(`--sampling first`). 영상의 뒤쪽은 읽지 않으므로 구도 확인용
타임랩스를 빠르게 만들 수 있습니다. 기본 출력은 24 FPS이므로 **입력 영상 하나가 출력 영상의 1초**가 됩니다.
없는 시각은 건너뛰고 실제 존재하는 영상만 연결합니다.

```bash
uv run python -m src.make_timelapse --device ANU-25-summer-3 --start-date 2026-07-01 --end-date 2026-07-05 --output bee_count_output/timelapse/device3_july.mp4

# 시·분·초까지 범위 지정
uv run python -m src.make_timelapse --device ANU-25-summer-3 --start-datetime "2026-07-05 06:00" --end-datetime "2026-07-05 18:00" --output bee_count_output/timelapse/device3_july05.mp4

# 영상 2개를 동시에 추출 (출력은 촬영 순서 유지)
uv run python -m src.make_timelapse --device ANU-25-summer-3 --start-date 2026-07-01 --end-date 2026-07-05 --workers 2

# 예전처럼 원본 전체를 고르게 샘플링하려면 명시적으로 uniform 선택
uv run python -m src.make_timelapse --device ANU-25-summer-3 --start-date 2026-07-01 --end-date 2026-07-05 --sampling uniform
```

좌측 상단에 `YYYY-MM-DD HH:MM:SS`와 기기명을 표시합니다. 캡션의 일시는
**파일명에 기록된 영상 시작 시각 + 추출 프레임 번호 / 원본 FPS**로 추정합니다.
카메라 시계나 파일명이 실제 촬영 시각과 다르면 캡션도 그 오차를 따릅니다.
원본 FPS를 알 수 없으면 파일명의 시각과 `video start; FPS unknown`을 표시합니다.
선택 기간은 파일명의 영상 시작 시각 기준(양끝 포함)이며, 선택한 영상 내부를
기간 끝에 맞춰 자르지는 않습니다. 날짜만 지정하면 종료일 하루 전체를 포함합니다.

주요 옵션:

- `--device`: 필수, 예: `ANU-25-summer-3`
- `--video-dir`: 원본 폴더, 기본 `videos`
- `--frames-per-video`: 영상당 출력 프레임 수, 기본 24
- `--sampling first|uniform`: 앞부분 연속 추출(기본) 또는 원본 전체 균등 추출. 모든 backend에 적용
- `--workers`: 동시에 프레임을 추출할 영상 수, 기본 1. 2 또는 4로 비교 가능
- `--fps`: 출력 FPS, 기본 24. 예를 들어 12로 지정하면 영상당 2초
- `--max-width`: 분석이 아닌 출력 이미지 최대 너비, 기본 1280
- `--output`: MP4 경로. 생략하면 `bee_count_output/timelapse/기기명_시작일시_종료일시.mp4`

첫 읽기 가능한 프레임의 비율로 출력 크기를 정하고, 다른 비율의 영상은 검은 여백을
추가해 원본 비율을 유지합니다. 오디오 없는 MP4와 같은 이름의 CSV를 생성합니다.
CSV에는 원본 영상, 출력 시작 위치, 프레임 수, 캡션 시각, 읽기 실패 여부를 기록합니다.
짧거나 일부 프레임을 읽지 못한 영상은 읽힌 프레임을 반복하여 영상당 출력 길이를
유지합니다. 전혀 읽지 못한 영상은 건너뛰고 콘솔과 CSV에 기록합니다.
최종 출력은 성공적으로 작성된 뒤 기존 출력 파일을 교체합니다.
병렬화는 프레임 추출에만 적용하며 캡션과 인코딩은 촬영 순서대로 진행합니다.
캐시는 사용하지 않습니다. 준비 중인 영상 수를 `--workers`로 제한하지만,
영상당 24장의 컬러 이미지를 보관하므로 작업 수에 따라 메모리 사용량도 증가합니다.
디스크 대역폭이나 디코더 CPU가 포화되면 작업 수를 늘려도 빨라지지 않을 수 있습니다.

### NVIDIA GPU에서 타임랩스 추출

Ubuntu의 NVIDIA GPU 환경에서는 `--backend cuda`로 FFmpeg의 하드웨어 디코딩과
`scale_cuda` 축소를 사용할 수 있습니다. 기존 CPU 방식은 `--backend opencv`가
기본입니다. 실행 환경에 정상 NVIDIA 드라이버와 CUDA 디코딩/`scale_cuda`를 지원하는
`ffmpeg`, `ffprobe` 실행 파일이 필요합니다. 먼저 `nvidia-smi`가 정상 동작해야 합니다.

```bash
uv run python -m src.make_timelapse --device ANU-25-summer-3 --start-date 2026-07-01 --end-date 2026-07-05 --backend cuda --workers 2 --output bee_count_output/timelapse/device3_gpu.mp4
```

`--gpu-device 0`이 기본 GPU이며 여러 GPU가 있으면 번호를 바꿀 수 있습니다.
GPU를 초기화하지 못하면 시작 단계에서 오류를 출력합니다. CPU로 자동 전환하지
않으며, 영상별 디코딩 오류는 CSV에 기록합니다. CSV의 `backend`로 실행 경로를
확인할 수 있습니다. GPU 경로는 8/10비트 YUV 4:2:0 입력을 지원하도록 구성했습니다.
지원하지 않는 픽셀 형식이나 프레임 수/FPS 메타데이터가 없는 영상은 CPU 경로를
사용해야 합니다. 캡션과 최종 MP4 인코딩은 CPU에서 실행합니다.

이 경로는 원본을 순차 디코딩하며, 선택한 프레임만 GPU에서 축소하고 CPU로 전달합니다.
기본 `--sampling first`는 앞 24프레임을 출력한 뒤 종료하며 전체 영상을 읽지 않습니다.
`--sampling uniform`은 마지막 샘플까지 원본 전체를 순차 디코딩합니다.
따라서 실제 속도는 원본 길이, 코덱, 디스크와 GPU 상태에
따라 달라지며 무조건 빨라지는 것은 아닙니다. 같은 기간으로 CPU/GPU 실행 시간을
비교하세요. `--backend ffmpeg`는 GPU 없이 같은 순차 추출 방식을 시험하는 옵션입니다.
모든 경로는 기본적으로 영상당 24프레임을 만들지만 축소 알고리즘과 디코더 차이로
픽셀이 완전히 같지는 않을 수 있습니다. GPU 경로는 회전 메타데이터를 적용하지 않습니다.

현재 개발 세션에서는 CUDA 초기화가 `CUDA_ERROR_NO_DEVICE`로 실패하여 RTX 3080
실행 속도와 실제 CUDA 필터 실행은 검증하지 못했습니다. CPU FFmpeg 추출, 프레임
선택/시각, CUDA 명령 구성과 GPU 사용 불가 처리에 대한 테스트를 수행했습니다.
구현 참고: [NVIDIA의 FFmpeg 가속 문서](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/ffmpeg-with-nvidia-gpu/index.html).

## 산출물

### 입구 자동 탐색 실험

`src/detect_entrance.py`는 **화면 아래 절반만** 탐색하며 빨간 구조물이나 특정
색상을 요구하지 않습니다. 영상 전체에서 균등 샘플링한 프레임의 중앙값 영상,
지속적인 수평 경계, 밝기 변화를 보정한 시간적 변화량으로 후보를 평가합니다.
ROI는 후보 주변에 여백을 추가하고 화면 아래 절반 안으로 제한합니다.

```bash
uv run python -m src.detect_entrance --videos videos/example.mp4 --samples 12
```

`bee_count_output/entrance_detection/`에 후보 사각형을 그린 JPG와 좌표/점수/상태를
담은 JSON을 저장합니다. 후보 점수는 확률이 아닙니다. 근거가 부족하거나 비슷한
후보가 여러 개면 `failed`로 기록하고 최종 좌표를 반환하지 않습니다.
단일 후보도 `needs_review`로 표시하며, 출력된 `--roi`/`--entrance`는 이미지에서
입구가 맞는지 확인한 후에만 사용해야 합니다. 입구 높이와 ROI 여백은 영상 높이에
비례한 휴리스틱이며 실제 개구부의 정확한 경계를 측정한 값은 아닙니다.

실제 영상 4개의 초기 시험에서 입구와 판재 경계를 혼동하는 사례를 확인했습니다.
따라서 이 도구는 **후보 탐색 시제품**이며 기존 카운팅의 좌표 선택에는 연결하지
않았습니다. 특히 벌이 없거나 가려진 장면에서도 동작하는 완전 자동 감지를 위해서는
입구/비입구 구분과 라벨 기반 검증이 추가로 필요합니다.

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

## 기간별 ROI / ENT 수동 편집기

`src/edit_roi.py`는 **기기와 적용 기간을 먼저 지정**한 뒤, 기간 안의 대표 영상 하나를 골라 ROI와 ENT를 편집하는 독립 도구입니다. 기간 내 다른 영상을 열어 볼 필요가 없습니다. 좌표는 기간 전체에 대한 YAML 기록으로 저장하며, 현재 벌 계수 코드의 설정이나 `roi_ent_info.txt`를 변경하지 않습니다.

GUI를 표시할 수 있는 데스크톱 터미널에서 실행합니다.

```bash
uv sync
uv run python src/edit_roi.py
```

기기 → 시작 일시 → 종료 일시 → 영상 번호 순서로 터미널에서 입력합니다. 기기는 `8` 또는 `ANU-25-summer-8`처럼 입력할 수 있습니다. 영상 목록은 20개씩 표시되며, 목록의 `n` / `b`는 다음/이전 페이지입니다. 전체 목록 기준 영상 번호를 입력하면 해당 영상의 첫 프레임을 표시합니다.

인수를 사용하면 반복 입력을 줄일 수 있습니다.

```bash
uv run python src/edit_roi.py \
  --device 8 \
  --start '2026-07-01 09:00:00' \
  --end '2026-08-09 18:00:00' \
  --output roi_regions.yaml
```

- 날짜만 입력하면 시작은 `00:00:00`, 종료는 `23:59:59`입니다. 시작/종료 모두 포함하며, **원본 파일명의 촬영 시작 일시**를 기준으로 적용 영상을 구분합니다. 시간대 변환은 하지 않습니다.
- 기본 영상 폴더는 `videos`이며 `--video-dir`로 변경합니다. `기기명_YYYYMMDD_HHMMSS.mp4` 형식의 원본 영상을 대상으로 합니다.
- `--video ANU-25-summer-8_20260705_120000.mp4`처럼 대표 영상의 파일명을 지정하면 목록 선택을 생략합니다. 선택한 기기/기간 안의 영상이어야 합니다.

편집 창에 키보드 포커스를 둔 상태에서 조작합니다.

| 키 | 동작 |
| --- | --- |
| Space | ROI 위 → 아래 → 왼쪽 → 오른쪽 → ENT 위 → 아래 → 왼쪽 → 오른쪽 순서로 선택 |
| 방향키 | 선택한 변을 수직 방향으로 원본 좌표 기준 1픽셀 이동 |
| Shift + 방향키 | 10픽셀 이동 |
| `n` / `b` | 다음 / 이전 프레임 표시. 편집 중인 좌표는 유지 |
| Enter | 편집 완료, YAML 저장 후 창 닫기 |
| Esc / 창 닫기 | 저장 없이 취소 |

ROI는 초록색, ENT는 하늘색, 선택한 변은 노란색입니다. ENT가 ROI를 벗어나거나 경계가 뒤집히는 이동은 제한합니다. 바깥쪽으로 늘릴 때는 ROI부터, 안쪽으로 줄일 때는 ENT부터 조절하면 됩니다. `--frame-step 24`를 주면 `n` / `b`가 한 번에 24프레임씩 이동합니다 (기본 1).

같은 기기의 **동일한 시작·종료 일시**로 실행하면 기존 YAML 좌표를 불러오고 해당 기록을 수정합니다. 처음 만드는 기간은 코드의 기기별 좌표 프리셋으로 시작하며, 유효한 프리셋이 없으면 화면 하단에 초기 박스를 표시합니다. 기존 기간과 일부만 겹치는 새 기록은 모호한 적용을 막기 위해 거부합니다. 기간 경계를 바꾸려면 YAML의 기존 시작/종료 일시도 함께 정리하세요.

저장 형식 예시 (숫자 배열은 한 줄로 기록):

```yaml
version: 1
regions:
- device: ANU-25-summer-8
  start: '2026-07-01 00:00:00'
  end: '2026-08-09 23:59:59'
  roi: [410, 980, 1640, 1232]
  entrance: [510, 1080, 1590, 1202]
  image_size: [1640, 1232]
  reference_video: ANU-25-summer-8_20260705_120000.mp4
  reference_frame: 24
```

좌표 순서는 원본 해상도의 `[x1, y1, x2, y2]`이며 오른쪽/아래 끝은 배열 슬라이스처럼 제외 경계입니다. `image_size`는 `[너비, 높이]`, `reference_frame`은 0부터 시작하는 프레임 번호입니다. 같은 기기라도 구도가 바뀐 시점부터 별도 기간으로 기록하면 됩니다. 실제 분석에는 아래의 `src/main.py --roi-yaml` 실행 방법으로 적용합니다.

### SSH 원격 접속에서 브라우저로 편집하기

원격 서버에서는 데스크톱 GUI 없이 웹 편집기를 사용할 수 있습니다. 외부 웹 라이브러리나 Node.js 설치는 필요하지 않습니다.

서버의 프로젝트 폴더에서 실행합니다 (종료: `Ctrl+C`).

```bash
uv run python src/roi_web.py --port 8000
```

로컬 PC의 터미널에서 포트 포워딩을 연결하고, 연결을 유지합니다.

```bash
ssh -N -L 8000:127.0.0.1:8000 사용자명@서버주소
```

로컬 브라우저에서 **http://localhost:8000**을 엽니다. VS Code Remote SSH를 사용한다면 Ports 탭에서 서버의 8000번 포트를 전달해도 됩니다. 서버는 `127.0.0.1`에만 바인딩됩니다.

1. 기기와 적용 시작·종료 일시를 입력하고 **영상 목록 보기**를 누릅니다.
2. 기간 안의 대표 영상을 고르고 **프레임 열기**를 누릅니다.
3. 편집 화면에 포커스를 두고 Space, 방향키, n/b로 경계와 프레임을 조절합니다. Tab 키로도 화면에 포커스를 옮길 수 있습니다.
4. Enter 또는 **저장** 버튼으로 서버의 `roi_regions.yaml`에 기록합니다. 저장 후 다른 기간을 계속 편집할 수 있습니다.

웹 화면의 경계선 이동은 브라우저에서 즉시 처리합니다. 프레임을 바꿀 때만 해당 프레임 이미지를 서버에서 받아 오므로 원본 영상을 내려받지 않습니다. 프레임을 이동해도 편집 좌표는 유지됩니다. Esc는 현재 영상의 편집 시작 또는 마지막 저장 시점의 좌표로 되돌립니다. 입력란에서는 편집 단축키가 적용되지 않습니다.

저장 형식과 기간 중복 검사, 기기별 초기 좌표는 데스크톱 편집기와 동일합니다. `--video-dir`와 `--output`으로 영상 폴더와 YAML 경로를 지정할 수 있습니다. 편집만으로 분석을 실행하지는 않으며, 저장한 YAML은 아래 batch 명령에서 지정해 적용합니다.

### YAML 영역을 적용해 기간별 옵티컬 플로우 일괄 실행

기존 `src/main.py --mode batch`의 영상 처리, 병렬 처리, CSV·프리뷰 출력을 그대로 사용합니다. `--roi-yaml`을 지정하면 각 **영상 파일명의 기기와 촬영 시작 일시**에 해당하는 YAML 기록의 ROI·ENT를 적용합니다. 같은 기기도 촬영 구도가 바뀐 날짜를 경계로 서로 다른 좌표가 자동 적용됩니다.

```bash
uv run python src/main.py --mode batch \
  --roi-yaml roi_regions.yaml \
  --devices 8 12 18 \
  --start-date 2026-07-01 --end-date 2026-08-31 \
  --workers 4 \
  --output-root bee_count_output --run-name yaml_jul_aug
```

기기는 번호 또는 전체 기기명을 사용합니다. 하나만 지정할 때는 `--device 8`도 가능합니다. 일시 단위로 범위를 정하려면 다음처럼 지정합니다.

```bash
uv run python src/main.py --mode batch \
  --roi-yaml roi_regions.yaml --device ANU-25-summer-8 \
  --start-datetime '2026-07-05 09:00:00' \
  --end-datetime '2026-08-10 18:00:00' \
  --output-root bee_count_output --run-name device8_period \
  --dry-run
```

`--dry-run`은 영상 파일을 열지 않고 파일명과 YAML만으로 선택된 영상과 적용 좌표를 확인합니다. YAML 좌표 형식·영역 관계는 검사하지만, 실제 영상 해상도와 읽기 가능 여부는 본 실행에서 검사합니다. 연산이나 산출물 저장은 하지 않습니다. 출력 내용을 확인한 뒤 `--dry-run`을 빼면 계산합니다. 날짜만 지정한 종료일은 23:59:59까지 포함하며, 양 끝 일시 모두 포함합니다. 영상 중간을 자르는 기능은 아니며, 범위 안에서 시작한 영상 전체를 처리합니다. 기존 `--start` / `--end`는 **영상 목록 인덱스**이므로 일시 선택에는 위의 `--start-date` / `--start-datetime`을 사용하세요.

- YAML 또는 기기·기간 필터를 지정하면 기본 탐색 패턴은 `*.mp4`입니다. 기존 7월 전용 패턴 때문에 8월 영상이 누락되지 않습니다. `--pattern`을 명시하면 추가로 그 패턴을 적용합니다.
- `--devices` 생략 시 폴더의 모든 기기가 대상이며, 기간의 한쪽 끝을 생략하면 그쪽은 제한하지 않습니다. 기존 `--videos`, `--limit`, 인덱스 범위도 함께 사용할 수 있습니다.
- 연산 시작 전 선택된 모든 영상의 YAML 매칭, 좌표, 해상도를 검사합니다. 해당 기기·촬영 일시의 좌표가 없으면 `[SKIP]`과 파일명·사유를 출력하고 해당 영상만 건너뛰며, 나머지 영상은 계속 분석합니다. 분석 대상과 건너뛴 영상 수도 출력합니다. 건너뛴 목록은 연산 시작 전에 `skipped_videos.csv`에도 저장하며, 완료 시 건수와 경로를 다시 출력합니다. 모든 영상에 좌표가 없어도 목록을 저장하고, 기존 분석 산출물은 변경하지 않고 종료합니다. 기간 중복이나 잘못된 좌표·해상도 불일치는 기존처럼 중단합니다.
- YAML 사용 시 `--roi`, `--entrance`, 명시적인 좌표 프리셋, `--reuse-existing`과의 혼용은 허용하지 않습니다. 옵티컬 플로우의 `--preset` 등 연산 설정은 기존처럼 사용합니다.
- **batch 실행은 YAML 사용 여부와 관계없이 완료된 결과를 검사해 재사용합니다.** 미완료·손상·설정 불일치 결과는 다시 계산하며, `--force`로 모든 영상을 강제 재계산할 수 있습니다. 같은 실행 폴더의 요약 CSV에는 재사용한 결과와 새 결과가 함께 기록됩니다.

위 첫 예시의 산출물은 `bee_count_output/yaml_jul_aug/`에 생성됩니다.

| 파일 | 내용 |
| --- | --- |
| `<영상명>_frame_flux.csv` | 프레임별 플로우 계수 결과 |
| `<영상명>_window_3sec.csv` | 3초 구간별 집계 |
| `<영상명>_preview.mp4` | 적용 영역과 계수 결과 프리뷰 |
| `batch_summary.csv` | 영상별 결과 및 실제 적용한 ROI·ENT 좌표 |
| `processing_timing.csv` | 영상별 연산 시간 |
| `skipped_videos.csv` | YAML 좌표가 없어 건너뛴 영상명·경로·기기·촬영 일시·YAML 경로·사유 |

`skipped_videos.csv`는 분석 전에 저장하므로 이후 분석 중 오류가 나도 남습니다. 같은 실행 폴더에서는 최신 실행 목록으로 갱신하고, 누락이 없으면 헤더만 저장합니다. 이전 목록도 보관하려면 다른 `--run-name`을 사용하세요. `--dry-run`에서는 이 파일을 생성하거나 변경하지 않습니다.


### 중단 후 batch 재실행 및 worker 변경

실행 중인 이전 프로세스가 종료된 것을 확인한 뒤, **같은 출력 폴더와 `--run-name`**으로 실행하면 완료된 영상을 재사용합니다. 이때 `--workers`만 늘려도 됩니다. 작업 중이던 영상은 처음부터 다시 계산하며, 프레임 중간부터 이어 계산하지는 않습니다. 같은 출력 폴더를 대상으로 두 실행을 동시에 돌리지는 마세요.

공통 재사용 검사는 batch와 tune의 영상별 작업에서 수행합니다. CSV의 필수 열·프레임 연속성·처리 길이·구간 집계 일치 여부와 프리뷰의 프레임 수·크기·FPS·첫/마지막 프레임 읽기를 검사합니다. 프리뷰 전체를 디코딩하는 손상 검사는 아닙니다. 예상 처리 길이는 현재 코드의 최대 2,880프레임 제한을 반영합니다. 프리뷰 간격이 영상보다 길어 의도적으로 프리뷰 프레임이 없는 경우에는 이미지 읽기 검사를 생략합니다.

새로 계산한 영상은 `<영상명>_completion.json`에 입력 영상의 경로·크기·수정 시각, 적용 설정, 결과 파일 해시와 영상별 요약을 기록합니다. 작업 시작 전에 미완료 상태를 기록하고, 결과 저장 및 검사를 모두 통과한 뒤에만 완료 상태로 바꿉니다. 중단된 기록이나 설정 변경, 파일 변경이 발견되면 다시 계산합니다. 입력 영상은 전체 해시 대신 경로·크기·수정 시각으로 식별합니다.

완료 기록이 없는 **기존 결과도 파일 검사를 통과하면 재사용**합니다. 다만 원래 적용한 좌표·설정을 확인할 수 없으므로 `reused_legacy_unverified`로 구분하고, 원래 좌표·설정 및 처리 시간은 요약에서 비워 둡니다. 집계 검증과 복원에는 현재 설정을 사용하므로 과거와 설정이 달라졌거나 과거 실행 결과가 섞였을 가능성이 있다면 `--force`로 다시 계산하세요. 기존 결과를 검증된 새 완료 기록으로 자동 승격하지 않습니다.

`batch_summary.csv`의 `result_status` 열과 마지막 터미널 요약에서 확인할 수 있습니다.

| 상태 | 의미 |
| --- | --- |
| `processed` | 이번 실행에서 계산·저장 완료 |
| `reused_verified` | 완료 기록의 입력·설정·파일 검사 통과 후 재사용 |
| `reused_legacy_unverified` | 완료 기록 없는 기존 파일 재사용, 원래 좌표·설정 확인 불가 |

`--dry-run`은 원본·결과 영상을 열지 않기 때문에 재사용 가능 여부까지 검사하지 않습니다. 실제 batch 실행 시 각 worker에서 검사하고, 통과한 영상은 옵티컬 플로우 계산을 생략합니다. `skipped_videos.csv`는 여전히 YAML 좌표 누락 목록이며, 완료 결과 재사용과 구분됩니다.
