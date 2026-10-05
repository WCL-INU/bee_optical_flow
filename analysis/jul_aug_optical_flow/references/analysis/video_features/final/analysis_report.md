# Video Feature Error Analysis Report

작성일: 2026-06-08

## 정리된 파일 구조

최종적으로 볼 파일은 `analysis/video_features/final` 아래에 모았다.

- `tables/low_flow_underprediction_summary.csv`: low-flow underprediction 조건별 요약
- `tables/low_flow_underprediction_feature_correlations.csv`: low-flow underprediction 오차와 feature 간 상관계수
- `tables/low_flow_underprediction_linear_context_summary.csv`: 선형회귀 context 차트 요약
- `tables/representative_feature_error_models.csv`: 대표 feature 조합 모델 성능
- `tables/representative_features.csv`: 강하게 중복되는 feature 그룹별 대표 feature
- `tables/feature_correlation_groups.csv`: feature 간 상관 그룹
- `tables/feature_error_correlations_outliers_iqr.csv`: outlier 제거 후 feature-error 상관계수
- `figures/low_flow_underprediction_scatter.png`: low-flow underprediction 산포도
- `figures/low_flow_underprediction_linear_context_chart.png`: 선형회귀식 위에 low-flow underprediction 정보를 얹은 차트
- `figures/representative_feature_error_model_scatter.png`: 대표 feature 조합 모델 산포도
- `figures/feature_error_scatter_outliers_iqr.png`: outlier 제거 후 주요 feature-error 산포도

원본 feature 파일은 `analysis/video_features/output`에 남겼고, 재현 가능하지만 최종 판단에 덜 중요한 중간 산출물은 `analysis/video_features/archive_intermediate_20260608`로 이동했다.

## 핵심 결론

전체 오차를 보면 `frame_diff_*` 계열이 가장 일관되게 강한 설명력을 보였다. 특히 `frame_diff_mean_p10`, `frame_diff_mean_min`, `frame_diff_changed_ratio_10_*` 등이 `sum_abs_error`, `total_abs_error`, `in_abs_error`, `out_abs_error`와 높은 상관을 보였다. 이는 단순한 공간 질감보다 시간축 변화량이 예측 오차와 더 직접적으로 연결된다는 의미다.

아웃라이어를 IQR 기준으로 제거하면 단순 frame diff 외에 방향성 관련 feature가 올라왔다. 대표적으로 `direction_balance_abs_mean`, `candidate_direction_entropy_p10` 등이 강해졌다. 즉 일부 큰 이상치가 제거된 뒤에는 움직임의 양뿐 아니라 방향 균형과 방향 분산도 오차와 관련이 있었다.

low-flow underprediction 조건에서는 결론이 더 뚜렷했다. 조건은 실제값이 0보다 크고, flow 값이 하위 25%이며, 예측값이 실제값보다 작은 경우로 잡았다. 이 조건에서 underprediction 오차는 `frame_diff_mean_p90`, `frame_diff_changed_ratio_10_p50`, `frame_diff_p90_p90` 등 시간축 변화 feature와 강하게 연결됐다.

low-flow underprediction 요약:

- `in_low_flow_under`: n=279, median under-error 14.94, p90 58.51
- `out_low_flow_under`: n=267, median under-error 13.71, p90 81.69
- `total_low_flow_under`: n=259, median under-error 28.87, p90 132.35

가장 중요한 해석은 다음과 같다. Optical flow 총량은 작게 잡혔지만 frame difference가 큰 경우, 실제 움직임 또는 영상 변화는 존재하는데 flow 기반 feature가 이를 충분히 반영하지 못해 underprediction이 커질 가능성이 높다.

## 질감 feature 해석

라플라시안 계열은 완전히 무의미하지는 않았다. 다만 핵심 설명 변수는 아니었다. low-flow underprediction에서는 `roi_laplacian_var_mean`이 음의 상관을 보였다. 즉 라플라시안 값이 낮은, 공간 질감이나 선명도가 부족한 영상에서 underprediction 오차가 커지는 경향이 있었다.

정리하면 다음과 같다.

- `frame_diff_*`: 시간축 변화량. low-flow underprediction에서 가장 중요하다.
- `roi_laplacian_var_*`: 한 프레임 내부의 공간 고주파, 선명도, 질감 정보. 보조적으로 의미가 있다.
- `roi_tenengrad_*`, `roi_edge_density_*`: 라플라시안보다 약하거나 일관성이 낮았다.

따라서 현재 가설은 `low flow + high frame diff + low texture` 쪽이 더 타당하다. 반대로 `low prediction + high frame diff + high laplacian` 조합은 엄격한 분위수 조건에서 충분한 샘플이 나오지 않아 핵심 가설로 유지하기 어렵다.

## Feature 중복과 대표 feature

전체 feature 795개 중 feature-feature pair는 315,615개였고, `|r| >= 0.95`인 매우 강한 중복 pair가 10,325개 있었다. 이를 그룹화하면 143개의 대표 feature 그룹으로 줄일 수 있었다.

가장 큰 그룹의 대표 feature는 `frame_diff_mean_p10`이었다. 이는 frame diff 계열과 주변 통계량들이 서로 매우 강하게 묶여 있다는 뜻이다. 후속 모델링에서는 전체 feature를 그대로 쓰기보다 대표 feature를 우선 사용하는 편이 낫다.

대표 feature 20개를 조합한 ridge 모델은 단일 feature보다 상관을 높였다.

- `sum_abs_error`: best single r=0.482, combined CV r=0.561, fitted r=0.581
- `total_abs_error`: best single r=0.462, combined CV r=0.544
- `in_abs_error`: best single r=0.458, combined CV r=0.532
- `out_abs_error`: best single r=0.457, combined CV r=0.525

즉 단일 feature보다 여러 대표 feature를 조합하는 것이 낫지만, 상승폭은 제한적이다. 현재 feature만으로 오차 원인을 완전히 설명하기보다는 위험 조건을 탐지하는 용도에 더 적합하다.

## 우선순위가 낮은 결과

공분산 결과는 feature scale의 영향을 크게 받으므로 최종 판단에서는 제외했다. 정규화한 공분산은 상관계수와 동일한 해석으로 귀결되므로 별도 결론으로 유지할 필요가 낮다.

전체 feature-feature correlation 원본 파일은 너무 크고 직접 해석하기 어렵다. 대신 대표 feature와 correlation group 결과만 최종 폴더에 남겼다.

`low prediction + high frame diff + high laplacian` 조건 차트는 샘플 수가 부족하거나 거의 비어 있어 핵심 결과에서 제외했다. 관련 파일은 archive로 이동했다.

## 다음 분석 방향

후속 분석은 모든 오차가 아니라 low-flow underprediction에 집중하는 것이 낫다. 우선 feature는 다음 범위로 제한하는 것을 권장한다.

- `frame_diff_mean_p90`
- `frame_diff_changed_ratio_10_p50`
- `frame_diff_p90_p90`
- `roi_laplacian_var_mean`
- `direction_balance_abs_mean`
- `candidate_direction_entropy_*`
- `total_filtered_traffic_flux`
- `total_filtered_to_raw_traffic_ratio`

모델링 목적은 실제 카운트를 직접 대체하는 것보다, optical flow 기반 예측이 낮게 잡힐 위험이 큰 구간을 탐지하고 보정하는 쪽이 더 현실적이다.
