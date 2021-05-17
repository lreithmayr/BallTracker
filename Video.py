import pathlib
import cv2
import os
from Tracker import ROITracker
from KalmanFilter import KalmanFilter
import imutils


if __name__ == "__main__":
    current_dir = pathlib.Path(__file__).parent.absolute()
    vid = os.path.join(current_dir, "videos/shot2.mp4")

    cap = cv2.VideoCapture(vid)
    tracker = ROITracker()
    dt = 1 / cap.get(cv2.CAP_PROP_FPS)
    kf = KalmanFilter(dt=dt)
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
            pred_pos = kf.estimate_position(pos[0], pos[1], frame)
            predictions.append((pred_pos[0], pred_pos[1]))
            kf.draw_estimation_error(frame)
            cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    tracker.plot_trace(predictions)
