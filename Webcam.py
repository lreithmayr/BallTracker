import cv2
import imutils
from KalmanFilter import KalmanFilter
from Tracker import ContourTracker


if __name__ == "__main__":
    vid = "http://192.168.0.94:8080/video"
    cap = cv2.VideoCapture(vid)
    tracker = ContourTracker()
    kf = KalmanFilter()
    predictions = []

    while True:
        check, frame = cap.read()
        frame = imutils.resize(frame, int(500), int(500))
        frame, radius = tracker.track_contours(frame)
        pos = tracker.get_center()
        cv2.imshow("Frame", frame)

        if tracker.center is not None:
            pred_pos = kf.estimate_position(pos[0], pos[1], frame, radius=radius)
            predictions.append(pred_pos)
            cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()