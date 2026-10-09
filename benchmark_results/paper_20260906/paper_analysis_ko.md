# Raspberry Pi 기반 벌통 입구 optical flow의 실시간 처리 가능성 평가

## 초록

본 실험은 Raspberry Pi에서 현재 구현된 dense Farneback optical flow 기반 벌통 출입량 계산이 녹화 영상의 원래 속도인 24 frame/s를 실시간으로 처리할 수 있는지 평가하였다. Raspberry Pi 카메라로 취득한 1640×1232, H.264, 24 FPS 영상 3개에서 실제 분석 ROI(1180×260)를 사용하였다. 각 영상의 초반·중반·후반에서 600 frame(25초) 구간을 층화 추출하고, 파일을 최대 속도로 처리하는 offline 모드와 24 Hz 입력 및 크기 1의 latest-frame queue를 사용하는 realtime 모드를 각각 평가하였다. 총 18개 구간 실행에서 5,291개의 처리 frame pair에 대한 단계별 지연을 기록하였다.

Offline 처리율은 평균 3.853 FPS였으며 end-to-end 지연은 평균 259.46 ms, p95 270.64 ms였다. 24 FPS의 frame budget인 41.67 ms에 대해 모든 측정 frame이 deadline을 초과하였다. Realtime 재생에서는 평균 처리율 3.398 FPS, end-to-end p95 381.29 ms, 전체 frame drop 85.69%를 기록하였다. Offline 평균 처리 시간 중 Farneback 계산이 84.93%를 차지하였다. 따라서 현재 구현과 ROI 조건에서는 24 FPS 실시간 처리가 불가능하며, 단순 입출력 최적화가 아니라 optical-flow 연산량을 최소 약 6배 줄이는 구조적 변경이 필요하다.

## 1. 연구 질문과 판정 기준

연구 질문은 “현재 코드와 현재 Raspberry Pi에서, 저장 영상과 동일한 24 FPS 입력을 frame backlog 또는 drop 없이 지속 처리할 수 있는가?”이다.

영상 주기가 `1000 / 24 = 41.67 ms`이므로 다음 기준을 측정 전에 고정하였다.

- 기본 실시간 기준: frame drop이 0이고 end-to-end latency p95가 41.67 ms 이하
- 배포 권장 기준: frame drop이 0이고 end-to-end latency p95가 33.33 ms 이하. 나머지 20%는 CSI/libcamera와 운영체제 변동을 위한 여유로 둔다.
- 평균 FPS만으로는 실시간성을 판정하지 않는다. 꼬리 지연, deadline miss 및 frame drop을 함께 사용한다.

## 2. 재료 및 방법

### 2.1 하드웨어 및 소프트웨어

| 항목 | 조건 |
|---|---|
| 보드/CPU | Raspberry Pi, ARM Cortex-A72 4 core, 최대 1.8 GHz |
| 메모리 | 약 2 GB |
| 운영체제 | Linux 6.6.74+rpt-rpi-v8, aarch64 |
| CPU governor | `ondemand` |
| Python | 3.11.2 |
| OpenCV | 4.13.0, Release, ARM NEON, pthreads |
| NumPy / pandas | 2.4.4 / 3.0.2 |
| OpenCV thread | 2 |
| OpenCL | 사용 불가 |

OpenCV thread 수는 동일 영상 125 frame pair를 이용한 사전 시험에서 결정하였다. 1, 2, 4 thread의 처리율은 각각 3.858, 3.945, 3.931 FPS였으며, 차이는 작았지만 2 thread가 가장 높은 처리율과 더 낮은 CPU 사용량을 보여 본 실험 조건으로 선택하였다. 이 사전 시험 결과는 본 실험의 통계에는 포함하지 않았다.

### 2.2 입력 영상

세 입력은 모두 Raspberry Pi 카메라로 취득한 H.264, YUV420p, 1640×1232, 24 FPS, 120초 영상이다.

| 영상 | 크기 |
|---|---:|
| `ANU-25-summer-1_20260702_120000.mp4` | 44,357,012 byte |
| `ANU-25-summer-6_20260716_130000.mp4` | 121,385,075 byte |
| `ANU-25-summer-6_20260716_140000.mp4` | 150,552,714 byte |

파일 크기 차이가 크므로 서로 다른 압축 복잡도와 장면 부하를 포함한다. 파일별 SHA-256은 `environment.json`에 기록하였다. 영상마다 frame 0, 1140, 2280에서 시작하는 600-frame 구간을 사용하여 초반·중반·후반을 고르게 포함하였다. 영상별 총 1,800 frame, 즉 전체 영상의 62.5%를 분석 입력으로 사용하였다.

### 2.3 Optical-flow 조건

실제 영상군에 기록된 ROI `(450, 970, 1630, 1230)`와 entrance rectangle `(530, 1070, 1580, 1200)`을 사용하였다. ROI 크기는 1180×260, 즉 306,800 pixel이다.

주요 처리 조건은 다음과 같다.

- Gaussian blur kernel: 5×5
- Farneback: pyramid scale 0.5, level 4, window 21, iteration 3, polynomial neighborhood 5, sigma 1.2
- flow magnitude threshold: 1.0
- normal-flow threshold: 0.5
- temporal persistence filter: decay 0.65, threshold 1.3
- connected-component area filter: 최소 200 pixel
- preview drawing 및 MP4 저장: 비활성화

Preview를 제외한 조건은 운영에 필요한 카운트 계산의 최선 조건이다. 따라서 이 조건에서 실패하면 preview를 추가한 전체 디버그 파이프라인도 실시간 기준을 만족할 수 없다.

### 2.4 실험 모드

Offline 모드에서는 OpenCV `VideoCapture`가 영상을 최대 속도로 읽고 모든 인접 frame pair를 처리하였다. 측정된 end-to-end latency에는 H.264 decode, ROI 전처리, Farneback, flux/filter 후처리가 포함된다.

Realtime 모드에서는 별도 producer가 원본 timestamp에 맞춰 24 Hz로 frame을 공급하였다. Queue 크기는 1이며, 소비자가 이전 frame을 처리하는 동안 새로운 frame이 도착하면 오래된 대기 frame을 폐기하고 최신 frame으로 교체하였다. 이는 지연이 무한히 누적되는 것을 막는 일반적인 live-camera 정책을 모사한다.

각 구간의 첫 48개 **처리된** frame pair는 warm-up으로 제외하였다. Offline에서는 구간당 551 pair가 최종 통계에 포함되었다. Realtime에서는 심한 frame drop으로 warm-up 이후 남은 표본 수가 구간당 29~47 pair였으며, 전체 332 pair가 통계에 포함되었다.

### 2.5 계측 및 통계

Python의 monotonic high-resolution `perf_counter_ns`로 다음 단계를 각각 측정하였다.

- decode
- crop, grayscale 변환 및 blur 전처리
- Farneback optical flow
- flux, persistence 및 component filter 후처리
- processor latency와 end-to-end latency

0.5초 간격으로 CPU frequency, SoC temperature 및 process RSS도 기록하였다. 결과는 영상 및 모드별 세 시간 구간의 평균±표준편차로 요약하였다. 95% 구간은 세 temporal segment를 단위로 10,000회 non-parametric bootstrap하여 산출하였으며 `aggregate_summary.csv`에 보존하였다. 이는 장면 구간 간 기술적 변동 범위이며, 여러 Raspberry Pi 표본에 대한 모집단 신뢰구간으로 해석해서는 안 된다.

## 3. 결과

### 3.1 영상별 실시간 성능

표의 값은 세 temporal segment의 평균±표준편차이다.

| 모드 | 영상 | 처리율 (FPS) | end-to-end p95 (ms) | deadline miss | drop | 기본 기준 |
|---|---|---:|---:|---:|---:|:---:|
| Offline | summer-1 12:00 | 3.915±0.025 | 265.09±2.68 | 100% | 0% | 실패 |
| Offline | summer-6 13:00 | 3.827±0.030 | 271.70±2.72 | 100% | 0% | 실패 |
| Offline | summer-6 14:00 | 3.817±0.021 | 271.52±2.81 | 100% | 0% | 실패 |
| Realtime | summer-1 12:00 | 3.604±0.195 | 347.24±14.05 | 100% | 84.89% | 실패 |
| Realtime | summer-6 13:00 | 3.299±0.120 | 381.84±22.88 | 100% | 86.00% | 실패 |
| Realtime | summer-6 14:00 | 3.292±0.176 | 388.00±28.51 | 100% | 86.17% | 실패 |

Offline 9개 구간을 합친 평균 처리율은 3.853 FPS로 요구 처리율의 16.1%에 불과하다. 동일 처리율을 24 FPS까지 높이려면 평균 throughput 기준 6.23배의 가속이 필요하다. Pooled offline p95는 270.64 ms로 frame budget의 6.50배였으며, realtime pooled p95는 381.29 ms로 budget의 9.15배였다.

Realtime producer는 총 5,400 frame을 공급하였고 그중 4,627 frame이 queue에서 교체되어 전체 drop rate는 85.69%였다. 실제 처리 frame 사이의 평균 간격은 원본 기준 7.03 frame, p95는 8 frame이었다. 따라서 현재 조건에서는 단순히 출력 FPS가 낮아지는 것뿐 아니라 인접-frame을 전제로 조정한 flow magnitude, persistence 및 flux calibration도 유지되지 않는다.

### 3.2 단계별 병목

| 단계 | Offline 평균 (ms) | Offline p95 (ms) | 평균 시간 비율 |
|---|---:|---:|---:|
| H.264 decode | 15.21 | 18.41 | 5.86% |
| 전처리 | 1.30 | 1.53 | 0.50% |
| Farneback flow | 220.36 | 227.00 | 84.93% |
| flux/filter 후처리 | 22.60 | 27.98 | 8.71% |
| 전체 processing (decode 제외) | 244.26 | 255.07 | — |
| End-to-end | 259.46 | 270.64 | 100% |

Farneback 단독 평균 시간이 220.36 ms이므로 H.264 파일 decode를 완전히 제거해도 41.67 ms 기준을 만족할 수 없다. Decode를 제외한 processing p95도 255.07 ms로 budget의 6.12배이다. 따라서 저장 파일과 CSI 입력 경로의 차이는 성능 수치를 일부 바꿀 수 있지만 실시간 실패 결론을 뒤집을 수 없다.

Realtime 모드에서는 producer decode와 consumer 계산이 동시에 CPU를 사용하면서 Farneback 평균이 252.24 ms로 증가하였고, 전체 processing 평균은 292.87 ms, end-to-end 평균은 331.21 ms였다.

### 3.3 자원 및 열적 안정성

- 전체 측정 중 최대 온도: 44.303°C
- 모든 실행에서 관측된 최소/평균 CPU frequency: 1.8 GHz
- 최대 process RSS: 276.42 MiB
- 평균 process CPU 사용량: offline 124.7%, realtime 248.2% (100%는 논리 core 1개)
- 평균 system CPU 사용량: offline 33.6%, realtime 65.1%

온도가 낮고 모든 표본에서 1.8 GHz가 유지되었으므로 관측된 저성능이 열에 따른 frequency 저하로 발생했다는 근거는 없다. 다만 현재 권한으로 `vcgencmd get_throttled`의 historical throttle bit를 읽지 못했으므로, “throttling이 절대 없었다”는 강한 표현 대신 “측정 frequency와 temperature에서 throttling 증거가 관찰되지 않았다”고 기술해야 한다.

## 4. 타당성 및 한계

1. 녹화 영상은 해상도, 장면, 움직임 및 압축 부하를 보존하지만 CSI sensor exposure, ISP, CSI 전송 및 libcamera latency는 재현하지 못한다.
2. Offline 파일 입력에는 H.264 decode가 포함되지만 CSI 카메라는 보통 ISP 처리된 YUV/RGB frame을 제공한다. 다만 decode를 제외한 core processing만으로도 deadline을 6배 이상 초과하므로 주 결론에는 영향이 없다.
3. 단일 Raspberry Pi만 측정했으므로 보드 간 제조 편차와 냉각 구성의 모집단 변동을 추론할 수 없다.
4. 각 영상의 62.5%를 층화 분석했으며 전 구간을 모두 사용하지 않았다. 관측된 구간 간 변동은 offline FPS 기준 약 1.3% 수준으로 작고 모든 frame이 deadline을 초과했지만, 전체 영상에 대한 표현은 이 표본 범위로 제한해야 한다.
5. Realtime 구간별 latency 표본이 29~47개로 작다. 그러나 332개 pooled 표본 모두 deadline을 초과했고 drop rate도 84% 이상이므로 통계적 불확실성이 통과/실패 판정을 바꿀 가능성은 낮다.
6. 본 실험은 계산 성능 평가이며 벌 출입 count의 정확도는 평가하지 않았다. 최적화 후에는 속도와 함께 기존 출력에 대한 수치적 동등성 또는 ground truth accuracy를 별도로 검증해야 한다.
7. 일반 Linux scheduler와 `ondemand` governor를 사용하였다. 이는 현재 배포 환경을 반영하지만 CPU affinity, realtime priority 및 background process를 완전히 통제한 실험은 아니다.

## 5. 결론

현재 dense Farneback 구현은 실제 ROI 1180×260과 24 FPS 조건에서 Raspberry Pi 실시간 처리가 불가능하다. Headless 최선 조건에서도 평균 처리율은 약 3.85 FPS이고 모든 측정 frame이 deadline을 초과했다. 24 Hz 입력을 모사하면 약 85.7%의 frame이 폐기되며, 계산되는 optical flow도 평균 약 7-frame 간격의 영상 사이에서 이루어져 현재 threshold와 flux calibration의 전제가 깨진다.

그러나 본 시스템의 실제 목적이 연속적인 24 FPS 실시간 출력이 아니라 **20분마다 2분 영상을 취득한 후 다음 촬영 전까지 분석하는 주기적 batch 운용**이라면 판단은 달라진다. 본 측정에서 2분 영상의 headless 분석 예상시간은 평균 약 12분 27초였고, 가장 느린 영상 기준으로도 약 12분 35초였다. 촬영 시작 시점을 기준으로 20분 주기를 적용하면 `2분 촬영 + 12분 35초 분석 + 5분 25초 여유`로 구성할 수 있다. 따라서 비스로틀링 조건에서는 다음 촬영 전에 분석이 완료되며 처리 backlog가 누적되지 않는다.

### 5.1 실제 주기 운용 가능성

24시간 연속 운용을 가정하면 하루 최대 72회의 촬영이 이루어지고, 총 촬영시간은 2시간 24분이다. 최악 측정시간을 적용한 총 분석시간은 약 15시간 6분이며, 나머지 약 6시간 30분은 주기 사이의 냉각·통신·파일 관리 여유로 사용할 수 있다. 이는 현재 알고리즘이 frame-level real time은 아니지만, 요구되는 20분 batch 주기에는 적용 가능한 처리량을 가진다는 것을 의미한다.

| 주기 구성 | 예상시간 |
|---|---:|
| 영상 촬영 | 2분 |
| Headless 분석 | 평균 12분 27초, 최악 약 12분 35초 |
| 다음 촬영까지의 최악 기준 여유 | 약 5분 25초 |
| 허용 가능한 분석시간 증가 | 약 43% |
| 이에 대응하는 허용 처리율 저하 | 약 30% |
| 하루 촬영 횟수 | 72회 |

다만 이 운용 가능 판정은 실내 측정과 동일하게 CPU가 1.8 GHz를 유지한다는 조건에 기반한다. Raspberry Pi 4의 권장 주변 온도 범위는 0~50°C이며, 공식 문서에 따르면 SoC가 80~85°C에 접근할 때 Arm core가 점진적으로 감속하고 85°C에서는 Arm core와 GPU가 모두 감속한다([Raspberry Pi 4 datasheet](https://datasheets.raspberrypi.com/rpi4/raspberry-pi-4-datasheet.pdf), [Raspberry Pi thermal documentation](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html)). 본 실내 시험에서는 최대 44.3°C와 1.8 GHz가 관찰되었지만, 밀폐 함체가 야외에서 직사광선을 받는 조건을 이 결과가 대표하지는 않는다.

현재 최악 분석시간 12분 35초가 사용 가능한 분석 구간 18분에 도달하는 지점은 처리시간이 약 43% 증가하거나 처리율이 약 30% 감소한 경우에 해당한다. 따라서 짧은 순간의 소폭 clock 감소는 주기 내에서 흡수할 수 있지만, 고온 또는 저전압으로 유효 처리율이 약 30% 이상 지속 저하되면 다음 촬영 전에 분석을 완료하지 못할 수 있다. 현장에서는 다음 조건을 운용 전제로 둔다.

- Raspberry Pi와 카메라를 직접 일사에서 차단하는 차양 및 흰색·고반사 외함 적용
- SoC 방열판과 함체 외부 방열부 사이의 열경로 확보, 필요시 온도 제어 팬 사용
- SoC, 함체 내부 및 외기 온도 동시 기록
- CPU temperature 70°C에서 경고, 75°C에서 분석 연기 또는 부하 축소, 80°C 접근 시 촬영·저장 우선 정책 적용
- `vcgencmd get_throttled`와 CPU frequency를 이용한 thermal throttling 및 undervoltage 이력 기록
- 분석시간이 17분을 넘거나 미처리 영상이 누적될 경우 장애 상태 기록
- 가장 더운 시간대와 무풍·직사광 조건에서 실제 함체로 최소 3시간 반복 시험 수행

여기서 70°C와 75°C는 제조사의 throttle 한계가 아니라 80°C 이전에 운용 여유를 확보하기 위한 프로젝트 수준의 보수적 기준이다. 현장 수용 기준은 모든 반복 주기에서 분석시간 18분 미만, backlog 0, thermal/undervoltage throttling 0으로 정의하는 것이 타당하다.

저장공간도 함께 고려해야 한다. 측정 영상의 2분당 파일 크기는 약 44~151 MB였으므로 원본을 모두 보존하면 하루 약 3.2~10.8 GB, 세 영상 평균을 적용하면 약 7.6 GB가 생성된다. 장기 운용에는 분석 완료 후 원본 순환 삭제, 선택 보존 또는 주기적 전송 정책이 필요하다.

종합하면, **차양과 방열을 적용하고 SoC 온도·주파수·저전압을 감시하여 비스로틀링 상태를 유지한다는 조건에서, 20분마다 2분을 촬영하고 해당 영상을 다음 촬영 전에 분석하는 운용은 현재 Raspberry Pi에서도 현실적으로 가능한 것으로 판단된다.** 다만 이는 연속 실시간 분석 가능성을 의미하지 않으며, 야외 함체를 사용한 반복 시험에서 분석시간 18분 미만과 backlog 0을 확인한 후 최종 현장 운용 가능 판정으로 확정해야 한다.

연속적인 24 FPS 처리까지 필요하다면 현재 코드를 그대로 CSI 카메라에 연결하는 것은 권장할 수 없다. 먼저 다음 순서로 최적화 실험을 수행해야 한다.

1. ROI 공간 downsampling 비율별 속도와 count 정확도의 Pareto curve 측정
2. 전체 ROI dense flow 대신 entrance boundary 주변 strip만 계산하도록 영역 축소
3. Farneback pyramid level, iteration, window 크기 축소에 따른 속도/정확도 grid 평가
4. sparse Lucas–Kanade 또는 block/gradient 기반 방향 flux와 비교
5. 카메라 ISP 또는 Picamera2 low-resolution stream에서 필요한 크기로 직접 입력하여 full-frame 변환 비용 제거
6. 24 FPS 기준을 만족한 후보에 대해서만 실제 CSI/Picamera2 sensor-to-result latency와 장시간 thermal test 수행

성능 목표는 평균만 24 FPS를 넘는 것이 아니라, 동일한 판정 기준인 p95 ≤ 41.67 ms, drop 0%를 만족하는 것이다. 배포 여유까지 고려하면 p95 ≤ 33.33 ms를 최종 목표로 유지하는 것이 타당하다.

## 6. 재현 자료

- `frame_timings.csv`: frame별 원시 단계 지연 및 deadline 판정
- `run_summary.csv`: 18개 구간별 요약과 자원 측정
- `aggregate_summary.csv`: 영상/모드별 평균, 표준편차 및 bootstrap 구간
- `pooled_summary.csv`: 모드별 pooled 기술통계와 요구 가속비
- `environment.json`: 실행 인자, 하드웨어/소프트웨어, Git commit, OpenCV build, 입력 SHA-256
- `figures/stage_breakdown.png`: 단계별 평균 지연
- `figures/latency_distribution.png`: end-to-end latency 분포
- `figures/realtime_latency_timeline.png`: realtime 시간축 지연

원시 자료 재계산 결과 CSV 간 frame count, p95 및 deadline miss가 모두 일치했으며 결측 cell과 `(mode, video, segment, frame)` 중복은 각각 0개였다.
