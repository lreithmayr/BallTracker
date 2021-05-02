import cv2
# from pykalman import KalmanFilter
import imutils
from Tracker import Tracker

if __name__ == "__main__":
    vid = "http://192.168.0.94:8080/video"
    cap = cv2.VideoCapture(vid)
    tracker = Tracker()

    while True:
        check, frame = cap.read()
        frame = imutils.resize(frame, int(1920 / 1.5), int(1080 / 1.5))
        tracked_frame = tracker.track_contours(frame)
        cv2.imshow("Frame", tracked_frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

    tracker.plot_track()
