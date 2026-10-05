# 영상 기반 벌통 모니터링 프로젝트 보고서 묶음

**기준일:** 2026-09-06

## 보고서 선택

| 문서 | 권장 독자 | 핵심 내용 |
|---|---|---|
| [프로젝트 현황·의의 보고서](./project_status_and_significance_ko.md) | 프로젝트를 처음 접하는 연구자, 개발자, 의사결정자 | 작동 원리, 전체 벤치마크, 현재 성숙도, 활용 가치, 향후 개발 과제 |
| [양봉 전문가용 보고서](./apiculture_expert_report_ko.md) | 양봉 연구자, 양봉가, 현장 시험 설계자 | 생물학적 해석 범위, 현장 경보, 적용 시나리오, 검증 SOP, 전문가 참여 지점 |

두 문서에는 Mermaid 흐름도와 기존 분석 PNG가 포함돼 있다. Markdown 뷰어에서 Mermaid를 지원하지 않더라도 표와 본문만으로 전체 결과를 읽을 수 있다.

## 한 문장 결론

> 현재 프로젝트는 정밀 자동 벌 계수기나 질병 진단기가 아니라, 알려진 설치 조건에서 벌통 입구의 방향성 활동을 주기적으로 수치화하고 일부 큰 과소계측을 품질 경보로 선별하며, Raspberry Pi 배치 운용 가능성까지 확인한 통합 연구 프로토타입이다.

## 대표 벤치마크

| 평가 축 | 핵심 결과 | 의미 |
|---|---|---|
| 최신 count 회귀 내부 적합 | IN R² 0.718, MAE 17.38; OUT R² 0.691, MAE 18.87 | flux와 수작업 count의 관계는 확인됐으나 독립 시험 성능은 아님 |
| 실제 총출입 ≥50의 총량 오차 | 절대상대오차 중앙값 23.7%, P90 69.8% | 중·고활동 추세는 유망하지만 큰 꼬리오차가 남음 |
| 고오차 과소계측 위험 분류 | 날짜 그룹 AUC 0.984/AP 0.854; 장치 전체 제외 AUC 0.835/AP 0.420 | 기존 장치 재검토 선별에는 유망, 새 장치 일반화는 부족 |
| Raspberry Pi offline | 3.853 FPS, p95 270.64 ms | 24 FPS 연속 실시간 처리 실패 |
| Raspberry Pi 배치 환산 | 2분 촬영 + 최악 약 12분 35초 분석 + 약 5분 25초 여유 | 20분 주기 배치의 계산상 가능성 확인 |

## 반드시 함께 읽을 해석 지침

1. 최신 회귀의 R²·MAE·RMSE는 같은 데이터에 적합하고 계산한 **내부 적합도**다. 독립 양봉장 또는 새 장치의 성능이 아니다.
2. 과거 분석의 R² 0.842/0.816 또는 0.773/0.748과 최신 0.718/0.691은 표본과 참값 0 처리 규칙이 달라 시간에 따른 성능 변화로 직접 비교하면 안 된다.
3. 실제 총출입 ≥50의 23.7%는 IN+OUT 합계의 오차다. IN 과대와 OUT 과소가 상쇄될 수 있으므로 방향별 정확도를 보장하지 않는다.
4. 신뢰도 모델의 성능은 **실제 총출입 ≥50임을 사후에 아는 모집단 안에서만** 성립한다. 현장에서는 참값을 알 수 없으므로 관측 가능한 적용 gate를 정하고 전체 스트림에서 전향 검증해야 한다.
5. 고오차 기준 56.2%는 오차분포 군집에서 유도된 값이지 양봉학적 허용 기준이 아니다.
6. 신뢰도 분류기만 날짜 교차검증됐고, count 회귀식·feature 선택·표적 및 임계값 결정까지 전부 fold 내부에서 수행한 완전한 end-to-end nested 검증은 아니다.
7. 20분마다 2분 촬영하는 방식은 10% duty-cycle의 **활동 표본**이다. 관측하지 않은 18분이나 하루 전체 출입 총계로 단순 외삽하면 안 된다.
8. Raspberry Pi 배치 가능성은 단일 실내 보드에서 녹화 H.264 영상을 처리한 속도의 환산 결과다. CSI 카메라의 sensor-to-result 지연, 야외 함체, 직사광, 장기 반복 운용은 아직 검증하지 않았다.
9. 영상 feature와 오차의 연관은 인과관계가 아니다. 고오차 원본을 양봉 행동 라벨과 함께 직접 판독해야 한다.
10. 최신 회귀·상대오차·신뢰도·장치별 분석은 현재 미커밋 예비 산출물이다. 논문 제출 전 데이터, 코드, 모델, 환경 및 Git commit을 동결하고 재생성해야 한다.

## 핵심 시각 자료

- [저활동 상대오차 분포](../analysis/nonzero_relative_error/output/relative_error_distribution.png)
- [참값 50 이상 방향별 특성 차이](../analysis/relative_error_actual50/output/direction_feature_differences.png)
- [고오차 과소계측 위험 모델](../analysis/underprediction_risk_model/output/model_validation.png)
- [장치 16·20 과대계측 진단](../analysis/device16_20_overprediction/output/overprediction_diagnostics.png)
- [Raspberry Pi 처리 단계별 병목](../benchmark_results/paper_20260906/figures/stage_breakdown.png)
- [Raspberry Pi latency 분포](../benchmark_results/paper_20260906/figures/latency_distribution.png)

## 원자료 출발점

- [현재 회귀 모델 비교](../validation/output/regression_model_comparison.csv)
- [참값 0 제외 상대오차 분석](../analysis/nonzero_relative_error/output/analysis_report.md)
- [참값 50 이상 오차 분석](../analysis/relative_error_actual50/output/analysis_report.md)
- [과소계측 위험 모델 분석](../analysis/underprediction_risk_model/output/analysis_report.md)
- [장치 16·20 원인 분석](../analysis/device16_20_overprediction/output/analysis_report.md)
- [Raspberry Pi 논문형 분석](../benchmark_results/paper_20260906/paper_analysis_ko.md)
