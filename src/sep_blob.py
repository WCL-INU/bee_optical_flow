import cv2
import numpy as np

video_path = "videos/ANU-25-summer-6_20260404_170000.mp4"
output_path = "videos/flow_blob_preview.mp4"

# ROI 좌표로 교체
x1, y1, x2, y2 = 1300, 1000, 1640, 1232

MOTION_THRESHOLD = 0.45

# 벌 크기에 맞게 조정
MIN_AREA = 50
MAX_AREA = 3000

MIN_W = 20
MIN_H = 20
MAX_W = 120
MAX_H = 120

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

open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

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
        levels=4,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    dx = flow[..., 0]
    dy = flow[..., 1]
    mag = np.sqrt(dx * dx + dy * dy)

    motion_mask = (mag > MOTION_THRESHOLD).astype(np.uint8) * 255

    # 핵심: motion fragment를 벌 크기 수준으로 재구성
    motion_mask = cv2.morphologyEx(
        motion_mask, cv2.MORPH_OPEN, open_kernel, iterations=1
    )
    motion_mask = cv2.dilate(motion_mask, dilate_kernel, iterations=1)
    motion_mask = cv2.morphologyEx(
        motion_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        motion_mask, connectivity=8
    )

    vis = roi.copy()
    count = 0

    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        bw = stats[label, cv2.CC_STAT_WIDTH]
        bh = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]

        if area < MIN_AREA or area > MAX_AREA:
            continue

        if bw < MIN_W or bh < MIN_H or bw > MAX_W or bh > MAX_H:
            continue

        aspect = bw / max(bh, 1)
        if aspect < 0.25 or aspect > 4.0:
            continue

        cx, cy = centroids[label]
        count += 1

        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 1)
        cv2.circle(vis, (int(cx), int(cy)), 3, (0, 0, 255), -1)

        cv2.putText(
            vis,
            str(count),
            (x, max(0, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        vis,
        f"count: {count}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    # 오른쪽 아래에 mask도 작게 표시하고 싶으면 사용
    mask_bgr = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
    small_mask = cv2.resize(mask_bgr, (w // 3, h // 3))
    vis[0 : h // 3, w - w // 3 : w] = small_mask
    
    writer.write(vis)
    prev_gray = gray

cap.release()
writer.release()

print(f"saved: {output_path}")
