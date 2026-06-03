# Validation Data Viewer

`data_viewer.html`은 optical-flow 산출값과 실제 카운트 값의 관계를 확인하기 위한 정적 브라우저 뷰어입니다. 빌드 단계에서 `merged_data.xlsx`를 읽고, 선형 회귀와 flat-exponential 회귀를 직접 계산한 뒤 데이터와 모델 결과를 HTML에 함께 넣습니다.

## 실행

PowerShell에서는 `uv` 사용을 기본으로 합니다.

```powershell
uv run python validation\build_data_viewer.py
```

생성 결과:

```text
validation/output/data_viewer.html
validation/output/regression_model_comparison.csv
```

## 폴더 구조

- `validation/build_data_viewer.py`: 현재 사용하는 뷰어 빌더
- `validation/data/`: 입력 spreadsheet와 검증 원천 데이터
- `validation/output/`: 재생성 가능한 뷰어 HTML과 모델 요약
- `validation/legacy/`: 이전 정적 그래프/리포트 스크립트와 산출물

기본 입력 파일은 `validation/data/merged_data.xlsx`입니다.

## 뷰어에서 확인할 것

- `IN` / `OUT` 방향별로 산출값과 실제값의 관계를 확인합니다.
- 모델 선택으로 `linear`와 `flat_exponential` 예측선을 비교합니다.
- `device`, `time` 필터로 특정 기기나 시간대의 오차 패턴을 봅니다.
- `Error percentile`로 큰 오차 기준을 조절합니다.
- 산점도 마커를 클릭하면 같은 데이터가 `Error Rows` 테이블에서 하이라이트되고 자동 스크롤됩니다.
- `video time`은 원본 `time` 코드에서 만든 표시값입니다. 예: `100000` -> `10:00:00`.
- `filtered`, `raw`, `ratio` 컬럼으로 개선 알고리즘이 실제 움직임은 유지하고 노이즈성 flux를 줄였는지 확인합니다.

## 지표 해석

뷰어는 선택한 모델로 `filtered flux`를 실제 count scale로 변환합니다.

```text
predicted = model(filtered flux)
error = predicted - actual
abs error = abs(error)
```

`error`가 양수면 모델이 실제보다 많이 예측한 것이고, 음수면 실제보다 적게 예측한 것입니다.

### 상단 요약 값

- `Rows`: 현재 `IN`/`OUT`, model, device, time 필터를 통과한 row 수입니다.
- `R squared`: 선택한 회귀 모델이 전체 데이터에서 실제 count 변동을 얼마나 설명하는지 나타내는 값입니다. 1에 가까울수록 모델 선이 데이터 관계를 잘 설명합니다.
- `Mean abs error`: 현재 필터된 row들의 `abs error` 평균입니다. 평균적으로 몇 마리 정도 빗나가는지 보는 값입니다.
- `P90 abs error`: 현재 필터된 row들의 `abs error`를 작은 값부터 큰 값까지 정렬했을 때 90번째 백분위 값입니다. 예를 들어 P90이 30이면, 현재 조건에서 약 90%의 row는 절대 오차가 30 이하이고 나머지 약 10%는 30보다 큽니다.
- `Large error rows`: 현재 `Error percentile` 기준 이상인 row 수입니다. 기본 percentile 90에서는 `abs error >= P90 abs error`인 row 수입니다.

`Error percentile` 슬라이더를 95로 올리면 더 극단적인 오차만 보게 되고, 50으로 낮추면 더 넓은 범위의 오차 row를 보게 됩니다.

### Large Error Features

이 영역은 `Large error rows`에 해당하는 row만 모아서 요약합니다.

- `Worst device`: 큰 오차 row 안에서 평균 `abs error`가 가장 큰 기기입니다. 특정 기기 ROI, 영상 조건, 카메라 위치 문제가 있는지 볼 때 유용합니다.
- `Worst time`: 큰 오차 row 안에서 평균 `abs error`가 가장 큰 시간대입니다. 특정 시간대의 조명, 벌 활동량, 그림자 조건을 의심할 수 있습니다.
- `Large-error actual mean`: 큰 오차 row들의 실제 count 평균입니다. 값이 높으면 고활동 구간에서 오차가 커진다는 뜻이고, 값이 낮으면 조용한 구간에서도 오차가 커진다는 뜻입니다.
- `Large-error raw flux mean`: 큰 오차 row들의 raw flux 평균입니다. 필터 적용 전 optical-flow 신호 규모입니다.
- `Large-error filtered flux mean`: 큰 오차 row들의 filtered flux 평균입니다. 필터 적용 후 모델 입력으로 남은 신호 규모입니다.
- `Filtered/raw ratio mean`: 큰 오차 row들의 `filtered / raw` 평균입니다. 큰 오차 구간에서 필터가 raw 움직임을 얼마나 남겼는지 봅니다.

### Error Rows 테이블

- `device`: 기기 번호입니다.
- `date`: 영상 날짜입니다.
- `video time`: 원본 `time` 코드를 시각 형태로 바꾼 값입니다.
- `actual`: 실제 수작업 count입니다.
- `pred`: 선택한 회귀 모델이 예측한 count입니다.
- `error`: `pred - actual`입니다. 양수는 과대추정, 음수는 과소추정입니다.
- `abs`: 오차 크기만 본 값입니다.
- `filtered`: 모델 입력으로 쓰인 filtered flux입니다.
- `raw`: 필터 적용 전 raw flux입니다.
- `ratio`: `filtered / raw`입니다. raw 움직임 신호 중 필터 이후에도 남은 비율입니다.

`ratio`는 필터 강도를 해석하는 보조 지표입니다. 1에 가까우면 raw 신호가 대부분 유지된 것이고, 0에 가까우면 필터가 대부분 제거한 것입니다. 단독으로 좋고 나쁨을 판단하기보다는 `actual`, `error`, `raw`, `filtered`와 함께 봐야 합니다.

예를 들어 `ratio`가 낮고 `actual`이 높은데 `error`가 크게 음수라면 실제 벌 움직임까지 필터가 과하게 제거했을 가능성이 있습니다. 반대로 `ratio`가 높고 `actual`이 낮은데 `error`가 크게 양수라면 노이즈성 움직임이 충분히 제거되지 않았을 가능성이 있습니다.

새 알고리즘 결과를 비교하려면 `validation/data/merged_data.xlsx`를 갱신한 뒤 빌드 명령을 다시 실행하고 `validation/output/data_viewer.html`을 새로고침하면 됩니다.
