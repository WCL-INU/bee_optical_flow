import cv2

video_path = "videos/ANU-25-summer-6_20260404_170000.mp4"

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

# 임시 좌표
x1, y1, x2, y2 = 1300, 1000, 1640, 1232
x3, y3, x4, y4 = 1400, 1200, 1600, 1232

# cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
# cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)
# cv2.imwrite("roi_check.jpg", frame)

# roi = frame[y1:y2, x1:x2]
# cv2.imwrite("roi_check.jpg", roi)

tmp = "080000"
video_path = f"bee_count_output/ANU-25-summer-6_20260405_{tmp}_entrance_count_preview.mp4"

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
for _ in range(330):
    ret, frame = cap.read()
roi = frame
cv2.imwrite(f"roi_check_entrance_count_{tmp}.jpg", roi)

# video_path = "bee_count_output/ANU-25-summer-6_20260405_140000_entrance_count_preview.mp4"

# cap = cv2.VideoCapture(video_path)
# ret, frame = cap.read()

# roi = frame
# cv2.imwrite("roi_check_entrance_count_140000.jpg", roi)
