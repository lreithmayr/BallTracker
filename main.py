from image_processing import *
import pathlib
import os
# from collections import deque

pts = []

if __name__ == "__main__":
    current_dir = pathlib.Path(__file__).parent.absolute()
    vid = os.path.join(current_dir, "shot.mp4")
    cap = cv2.VideoCapture(vid)
    initBB = None
    tracker = cv2.TrackerCSRT_create()
    while True:
        check, frame = cap.read()
        frame
        tracker, initBB = track_roi(tracker, frame, initBB, pts)
        if cv2.waitKey(1) == 27:
            sys.exit()
    cap.release()
    cv2.destroyAllWindows()
