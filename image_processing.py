import cv2
import sys
import numpy as np

def track_roi(tracker, frame, initBB, pts):
    if frame is None:
        sys.exit()
    if initBB is not None:
        (success, box) = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            x_center = int(x + (w / 2))
            y_center = int(y + (h / 2))
            cv2.circle(frame, (x_center, y_center), 0, (0, 255, 0), thickness=2)
            pt = (x_center, y_center)
            pts.append(pt)
            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None:
                    continue
                thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (0, 255, 0), thickness)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(20) & 0xFF == ord("s"):
        initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        tracker.init(frame, initBB)
    if cv2.waitKey(1) == 27:
        sys.exit()
    return tracker, initBB