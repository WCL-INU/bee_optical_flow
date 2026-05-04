import cv2
import numpy as np

video_path = "videos/ANU-25-summer-6_20260404_170000.mp4"
output_path = "videos/flow_arrow_preview.mp4"

# 이미 설정한 ROI 좌표로 교체
x1, y1, x2, y2 = 1300, 1000, 1640, 1232

# 화살표 간격
step = 16

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

    vis = roi.copy()

    for y in range(0, h, step):
        for x in range(0, w, step):
            dx, dy = flow[y, x]

            mag = np.sqrt(dx * dx + dy * dy)

            # 작은 움직임은 표시하지 않음
            if mag < 0.7:
                continue

            start = (x, y)
            end = (int(x + dx * 4), int(y + dy * 4))

            cv2.arrowedLine(vis, start, end, (0, 255, 0), 1, tipLength=0.3)

    writer.write(vis)

    prev_gray = gray

cap.release()
writer.release()

print(f"saved: {output_path}")
