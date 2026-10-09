# ROI 크기에 따른 Farneback 계산시간 분석

생성 시각: 2026-10-09T10:11:23.469039+00:00

## 초록 및 핵심 판정

본 시험은 Raspberry Pi 4에서 YAML로 지정한 네 종류의 실제 ROI와 동일 장면을 다섯 크기로 축소한 통제 ROI를 사용하여 Farneback 계산시간의 면적 의존성을 평가하였다. 9개 영상의 초·중·후반에서 총 72개 run, 39,672개 frame pair를 측정하였다.

통제 실험에서 ROI 100,000픽셀 증가당 평균 Farneback 시간은 74.35 ms 증가했고 (paired-cluster bootstrap 95% 구간 74.10~74.59 ms), log-log exponent는 1.125였다. 따라서 이 구현에서는 ROI 픽셀 수가 계산시간을 지배하는 1차 요인이며, 큰 배열에서 처리량이 다소 낮아져 정확한 선형 비례보다 약간 가파르게 증가한다.

Farneback 단독 p95의 24 FPS 경계는 약 77,161픽셀, 20% 여유 경계는 약 66,328픽셀로 추정되었다. 그러나 decode·전처리·runtime resize까지 포함하면 가장 작은 53,856픽셀 조건도 평균 20.0 FPS였으므로, flow-only 통과를 sensor-to-result 실시간 통과로 해석해서는 안 된다.

가장 큰 실제 ROI(1602×273, 437,346픽셀)에 대해 count 후처리까지 포함한 별도 full-pipeline 교차시험을 수행한 결과 평균 2.688 FPS, end-to-end p95 387.01 ms였다. 2분 영상 분석시간은 평균 17.85분, 관측 최악 17.90분이었다. 촬영 시작 간격이 20분이면 2분 촬영 후 남는 최악 여유는 6.2초뿐이며, 처리시간이 약 0.57%만 증가해도 backlog가 발생한다.

따라서 **최대 ROI를 그대로 사용하는 20분 시작 주기는 이번 실내 조건에서 수치상 간신히 완료됐지만, 충분한 운용 여유가 없으므로 야외 현장 운용 가능으로 판정할 수 없다.** 최적화 전에는 촬영 시작 간격을 최소 25분으로 늘리거나, ROI/downsampling으로 연산량을 줄인 뒤 full-pipeline과 정확도를 다시 검증해야 한다.

## 실험 설계

- 입력 영상: 9개
- YAML: `roi_regions.yaml`
- 실제 ROI 실험: True
- 동일 장면 통제 실험: True
- 통제 실험 기기: `ANU-25-summer-14`
- scale: 0.35, 0.5, 0.65, 0.8, 1.0
- 영상당 temporal segment: 3개
- segment 길이: 600 frames
- 제외한 warm-up: 48 frame pairs
- OpenCV threads: 2
- job 실행 순서: seed 20261009로 무작위 배치
- bootstrap: 5,000회, video×temporal-segment paired cluster 단위
- `flow_ms`는 `cv2.calcOpticalFlowFarneback` 호출만 포함하며 decode, resize, grayscale/blur는 제외하였다.

## 실제 YAML ROI 결과

| 기기 | ROI | 픽셀 수 | 영상 | 구간 | 평균 flow (ms) | 구간 p95 평균 (ms) | Flow-only FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| ANU-25-summer-1 | 1229×252 | 309,708 | 1 | 3 | 207.16 | 212.42 | 4.83 |
| ANU-25-summer-14 | 1602×273 | 437,346 | 3 | 9 | 313.65 | 324.93 | 3.19 |
| ANU-25-summer-16 | 773×245 | 189,385 | 3 | 9 | 114.95 | 118.97 | 8.70 |
| ANU-25-summer-6 | 1236×260 | 321,360 | 2 | 6 | 230.75 | 236.70 | 4.33 |

실제 ROI 비교에는 기기, 장면, 종횡비 차이가 함께 포함되므로 인과적 면적 효과는 아래의 동일 장면 통제 실험을 우선해 해석한다.

## 동일 장면 크기 통제 결과

| Scale | ROI | 픽셀 수 | 구간 | 평균 flow (ms) | 구간 p95 평균 (ms) | MPixel/s |
|---:|---:|---:|---:|---:|---:|---:|
| 0.35 | 561×96 | 53,856 | 9 | 29.54 | 30.40 | 1.823 |
| 0.50 | 801×136 | 108,936 | 9 | 65.23 | 67.38 | 1.670 |
| 0.65 | 1041×177 | 184,257 | 9 | 110.43 | 113.96 | 1.669 |
| 0.80 | 1282×218 | 279,476 | 9 | 189.15 | 194.66 | 1.478 |
| 1.00 | 1602×273 | 437,346 | 9 | 312.60 | 323.56 | 1.399 |

## 회귀 결과

- 평균 계산시간은 ROI 100,000픽셀 증가당 74.35 ms 증가했다 (paired-cluster bootstrap 95% 구간 74.10~74.59 ms).
- 평균 시간 선형모델 R²: 0.9968
- 평균 시간 log-log exponent: 1.125 (95% 구간 1.123~1.126); 1에 가까울수록 픽셀 수에 거의 비례한다.
- p95 시간 선형모델 R²: 0.9966
- 평균 Farneback 41.67 ms 예측 픽셀 수: 78642 (관측 범위 내: True)
- p95 Farneback 41.67 ms 예측 픽셀 수: 77161 (관측 범위 내: True)
- p95 Farneback 33.33 ms 예측 픽셀 수: 66328 (관측 범위 내: True)

선형모델의 음의 절편은 가장 작은 관측 ROI 아래에서 물리적 의미가 없으므로 모델과 예측은 53,856~437,346픽셀의 관측 범위 안에서만 사용한다.

## 부분 파이프라인과 2분 영상 환산

아래 시간은 decode, 필요 시 resize, grayscale/blur 및 Farneback까지만 포함하고 count 후처리를 제외한 **낙관적 하한**이다. 실제 프로그램 운용시간으로 사용하지 않는다.

| 기기 | ROI 픽셀 | 부분 파이프라인 FPS | 2분 영상 하한 (분) |
|---|---:|---:|---:|
| ANU-25-summer-16 | 189,385 | 7.698 | 6.23 |
| ANU-25-summer-1 | 309,708 | 4.432 | 10.83 |
| ANU-25-summer-6 | 321,360 | 4.020 | 11.94 |
| ANU-25-summer-14 | 437,346 | 3.004 | 15.97 |

## 최대 ROI full-pipeline 교차시험과 운용 주기

`ANU-25-summer-14_20260816_140000.mp4`의 초·중·후반 3개 구간에서 실제 count 후처리까지 수행하였다.

| 지표 | 결과 |
|---|---:|
| Full-pipeline 처리율 평균 | 2.688 FPS |
| Full-pipeline 처리율 관측 최저 | 2.681 FPS |
| End-to-end 평균 / p95 | 371.92 / 387.01 ms |
| Decode / 전처리 / Farneback / 후처리 평균 | 15.70 / 1.54 / 314.64 / 40.04 ms |
| 2분 영상 분석시간 평균 / 관측 최악 | 17.85 / 17.90분 |
| 20분 시작 주기에서 촬영 포함 여유 평균 / 최악 | 8.9 / 6.2초 |

이 교차시험은 가장 큰 ROI의 1개 영상과 세 시간 구간에 대한 기술적 반복이다. 20분 주기의 최악 환경 보증값이나 여러 Raspberry Pi의 모집단 추정치는 아니다. 오히려 관측 최악 여유가 6초에 불과하므로, 부팅·파일 flush·통신·CSI 입력 차이·고온·저전압을 흡수할 수 없다는 실패 여유 분석으로 해석한다.

## 열적 안정성과 야외 운용 조건

ROI 크기 시험의 구간별 최고 SoC 온도는 45.3°C였고, 평균 CPU frequency는 1797.6~1800.0 MHz였다. ROI 크기를 보정한 계산시간 residual의 실행 순서 기울기는 job당 0.0070 ms로 작았다.
Full-pipeline 교차시험의 최고 온도는 44.3°C였고 세 구간 모두 평균/최소 CPU frequency가 1800/1800 MHz였다. 따라서 이번 실내 시험에서는 지속적인 thermal frequency 저하나 시간 드리프트의 증거가 관찰되지 않았다.

다만 `vcgencmd get_throttled`는 `/dev/vcio` 권한 문제로 읽지 못했으므로 과거 thermal/undervoltage bit가 없었다고 단정할 수 없다. 또한 본 시험은 직사광선과 밀폐 함체의 복사열을 재현하지 않았다. Raspberry Pi 4 공식 데이터시트의 권장 주변온도는 0~50°C이며 지속 고부하·고온에서는 추가 냉각이 필요할 수 있다. 공식 문서는 core가 80~85°C 사이에서 점진적으로 감속하고 85°C에서 Arm과 GPU를 감속한다고 설명한다([Pi 4 datasheet](https://datasheets.raspberrypi.com/rpi4/raspberry-pi-4-datasheet.pdf), [Raspberry Pi thermal documentation](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#frequency-management-and-thermal-control)).

현장 배포 전제는 다음과 같다.

- 보드와 카메라를 직접 일사에서 차단하고, 고반사 외함·방열판·외함까지 이어지는 열경로와 필요 시 온도 제어 팬을 사용한다.
- SoC·함체 내부·외기 온도, CPU 실제 주파수, `vcgencmd get_throttled`, 저전압 및 각 영상 분석 완료시간을 주기별로 기록한다.
- 프로젝트 보호 기준으로 70°C 경고, 75°C에서 분석 연기/부하 축소, 80°C 접근 시 촬영·저장을 우선한다. 70/75°C는 제조사 throttle 한계가 아니라 사전 대응 기준이다.
- 가장 더운 무풍·직사광 조건에서 실제 함체와 CSI 카메라로 최소 3시간 반복하고, backlog 0 및 thermal/undervoltage event 0을 수용 기준으로 둔다.
- 최대 ROI를 유지한다면 20분 대신 최소 25분 시작 주기를 권장한다. 20분을 유지하려면 full-pipeline 분석시간 목표를 16분 이하로 두고 ROI/downsampling을 최적화한 뒤 재측정한다.

## 구현 및 ROI 최적화 해석

- Python이 전체 실행을 조정하지만 Farneback 자체는 OpenCV의 컴파일된 C++ 구현에서 수행된다. 최대 ROI full-pipeline 평균의 약 84.6%가 Farneback이므로 C++/Rust 재작성만으로 큰 폭의 개선을 기대하기 어렵다. 재작성은 Python 객체 생성·호출·복사 비용을 줄일 수 있으나, 우선순위는 픽셀 수·pyramid level·iteration 및 알고리즘 변경이다.
- 0.8 scale은 1.0 대비 픽셀 수가 36.1% 줄고 Farneback 평균시간은 39.5% 감소했다. 20분 주기 여유 확보 후보지만, runtime resize와 후처리를 포함한 full-pipeline 및 counting 정확도 시험을 통과해야 한다.
- 입구 경계의 얇은 strip만 바로 optical flow에 입력하면 큰 변위나 pyramid/window 문맥이 strip 밖으로 나가 대응점 추정이 불안정할 수 있다. strip 전략은 경계 양쪽에 예상 최대 변위와 Farneback support를 포함한 padding을 두고, 넓은 padded ROI에서 flow를 계산한 뒤 중앙 counting band만 평가하는 방식으로 검증한다.
- Picamera2 low-resolution stream 또는 필요한 크기의 입력을 직접 받으면 runtime resize 비용을 줄일 수 있지만, 저장 H.264 decode와 CSI/ISP 경로가 다르므로 실제 sensor-to-result 시험이 필요하다.

## 해석상의 한계

- 실제 YAML ROI 결과는 자연 장면 비교이므로 ROI 면적 외에 장면과 종횡비가 다르다.
- 통제 실험은 동일 ROI를 resize하므로 내용과 시간 구간은 짝지어지지만 interpolation, blur 및 noise 통계가 변한다. 이는 센서가 작은 해상도로 직접 취득한 영상과 완전히 동일하지 않다.
- 프레임은 독립 표본이 아니므로 신뢰구간은 frame 단위가 아니라 video×temporal-segment paired cluster를 재표본화했다.
- 신뢰구간은 한 Raspberry Pi에서의 영상·시간 구간 변동을 나타내며 보드 모집단의 신뢰구간이 아니다.
- 본 결과는 계산시간 분석이며 downsampling 후 optical-flow 및 count 정확도 보존을 입증하지 않는다.
- 녹화 H.264는 장면과 해상도를 보존하지만 CSI exposure, ISP, libcamera 전달 지연과 야외 열환경은 재현하지 않는다.

## 재현 자료

- `frame_timings.csv`: frame pair별 원시 계측
- `run_summary.csv`: video×segment×scale 요약
- `roi_size_summary.csv`: native/controlled 크기별 집계
- `regression_results.csv`: 선형 및 log-log 모델과 bootstrap 구간
- `regression_predictions.csv`: 관측 크기별 모델 예측
- `environment.json`: 실행 인자, 환경, 입력 영상 및 YAML hash
- `figures/`: 300 DPI 그래프
- `full_pipeline_crosscheck/`: 최대 ROI의 실제 count 후처리 포함 3구간 교차시험
