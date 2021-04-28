import cv2
import sys
from collections import deque

def track_roi(tracker, frame, initBB):
    centroids = deque(maxlen=64)
    if initBB is not None:
        (success, box) = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            print(x, y, w, h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            x_center = int(x + (w / 2))
            y_center = int(y + (h / 2))
            cv2.circle(frame, (x_center, y_center), 1, (0, 255, 0), thickness=2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(50) & 0xFF == ord("s"):
        initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        tracker.init(frame, initBB)
    if cv2.waitKey(1) == 27:
        sys.exit()
    if frame is None:
        sys.exit()
    return tracker, initBB