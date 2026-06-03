# Validation Data Viewer

`data_viewer.html`은 optical-flow 산출값과 실제 카운트 값의 관계를 확인하기 위한 정적 브라우저 뷰어입니다. 빌드 단계에서 `merged_data.xlsx`를 읽고, 선형 회귀와 flat-exponential 회귀를 직접 계산한 뒤 데이터와 모델 결과를 HTML에 함께 넣습니다.

## 실행

PowerShell에서 `uv` 사용을 기본으로 합니다.

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
- 산점도 마커를 클릭하면 같은 데이터가 `Error Rows` 테이블에서 하이라이트되고 자동 스크롤됩니다.
- `video time`은 원본 `time` 코드에서 만든 표시값입니다. 예: `100000` -> `10:00:00`.
- `filtered`, `raw`, `ratio` 컬럼으로 개선 알고리즘이 실제 움직임은 유지하고 노이즈성 flux를 줄였는지 확인합니다.

새 알고리즘 결과를 비교하려면 `validation/data/merged_data.xlsx`를 갱신한 뒤 위 빌드 명령을 다시 실행하고 `validation/output/data_viewer.html`을 새로고침하면 됩니다.
