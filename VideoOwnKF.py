import pathlib
import cv2
import os
from Tracker import ROITracker
from KFFromScratch import KF
import imutils
import numpy as np


if __name__ == "__main__":
    current_dir = pathlib.Path(__file__).parent.absolute()
    vid = os.path.join(current_dir, "videos/shot2.mp4")

    cap = cv2.VideoCapture(vid)
    tracker = ROITracker()

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    dt = 1 / fps
    kf = KF(dt)

    predictions = []

    while True:
        check, frame = cap.read()
        if not check:
            break

        frame = imutils.resize(frame, 1000, 500)
        frame_nr = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cv2.imshow("Frame", frame)

        if frame_nr == 1:
            tracker.set_init_bb(cv2.selectROI("Frame", frame, fromCenter=False,
                                              showCrosshair=True))
            tracker.init_roi_tracker(frame)

        if tracker.init_bb is not None:
            pos = tracker.track_roi(frame)
            x_updt, p_updt = kf.predict(np.array([pos[0], pos[1]]).reshape(2, 1))
            cv2.circle(frame, (int(x_updt[0]),int( x_updt[1])), 15, [0, 20, 255], 2, 8)
            #  pos = x_updt_i
            # predictions.append((pred_pos[0], pred_pos[1]))
            # kf.draw_estimation_error(frame)
            cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # tracker.plot_trace(predictions)
