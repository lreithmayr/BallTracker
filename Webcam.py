import cv2
import imutils
from Tracker import Tracker

if __name__ == "__main__":
    vid = "http://192.168.0.94:8080/video"
    cap = cv2.VideoCapture(vid)
    tracker = Tracker(cap)

    while True:
        check, frame = cap.read()
        frame = imutils.resize(frame, int(500), int(500))
        tracked_frame = tracker.track_contours(frame)
        pos = tracker.get_position()
        vel = tracker.get_velocity()
        print(vel[-1])
        cv2.imshow("Frame", tracked_frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

    tracker.plot_track()