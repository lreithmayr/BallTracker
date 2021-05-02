import pathlib
import cv2
import os
from Tracker import Tracker
import imutils

if __name__ == "__main__":
    current_dir = pathlib.Path(__file__).parent.absolute()
    vid = os.path.join(current_dir, "shot2.mp4")

    cap = cv2.VideoCapture(vid)
    frame_nr = cap.get(cv2.CAP_PROP_POS_FRAMES)
    num_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    tracker = Tracker()

    while True:
        check, frame = cap.read()
        if frame is not None:
            frame = imutils.resize(frame, 1000, 500)
            tracked_frame = tracker.track_contours(frame)
            cv2.imshow("Frame", tracked_frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
