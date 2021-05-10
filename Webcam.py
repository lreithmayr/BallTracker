import cv2
import imutils
from KalmanFilter import KalmanFilter
from Tracker import Tracker


if __name__ == "__main__":
    vid = "http://192.168.0.94:8080/video"
    cap = cv2.VideoCapture(vid)
    tracker = Tracker(cap)
    kf = KalmanFilter()

    while True:
        check, frame = cap.read()
        frame = imutils.resize(frame, int(500), int(500))
        cv2.imshow("Frame", frame)
        frame, pos = tracker.track_contours(frame)  

        if tracker.center is not None:
            pred_pos = kf.estimate_position(pos[0], pos[1])
            cv2.circle(frame, (pred_pos[0], pred_pos[1]), 20, [0, 60, 255], 2, 8)
            cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.plot_track()