import cv2
import numpy as np

video_path = "videos/ANU-25-summer-6_20260404_170000.mp4"
output_path = "videos/flow_mag_preview.mp4"

# 이미 설정한 ROI 좌표로 교체
x1, y1, x2, y2 = 1300, 1000, 1640, 1232

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 24.0

ret, frame = cap.read()
if not ret:
    raise RuntimeError("비디오를 읽을 수 없습니다.")

roi = frame[y1:y2, x1:x2]
prev_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

h, w = prev_gray.shape

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    dx = flow[..., 0]
    dy = flow[..., 1]
    mag = np.sqrt(dx * dx + dy * dy)

    # 작은 움직임 제거
    mag[mag < 0.5] = 0

    # 보기 좋게 정규화
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag_u8 = mag_norm.astype(np.uint8)

    heatmap = cv2.applyColorMap(mag_u8, cv2.COLORMAP_JET)

    # 원본 ROI와 섞어서 표시
    overlay = cv2.addWeighted(roi, 0.6, heatmap, 0.4, 0)

    writer.write(overlay)

    prev_gray = gray

cap.release()
writer.release()

print(f"saved: {output_path}")
