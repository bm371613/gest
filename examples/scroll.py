import argparse

import cv2
import pynput.mouse

from gest.cv_gui import text
from gest.inference import InferenceSession
from gest.math import relative_average_coordinate, accumulate

parser = argparse.ArgumentParser()
parser.add_argument("model_file", help="Model file")
parser.add_argument("--camera", help="Camera index", type=int, default=0)
parser.add_argument("--sensitivity", help="Scrolling sensitivity", type=int, default=10)


class App:

    def __init__(self, camera, model_file, scrolling_sensitivity):
        self.camera = camera
        self.inference_session = InferenceSession(model_file)
        self.mouse = pynput.mouse.Controller()

        self.scrolling_sensitivity = scrolling_sensitivity

    def run(self):
        capture = cv2.VideoCapture(self.camera)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        acc_score = None
        acc_relative_y = None
        root_relative_y = None
        on = False
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            heatmap = self.inference_session.cv2_run(frame).max(axis=0)

            # scrolling state on/off
            score = heatmap.max()
            acc_score = accumulate(acc_score, score, accumulated_weight=2)
            if on and acc_score < .3:
                on = False
                acc_relative_y = None
                root_relative_y = None
            if not on and acc_score > .5:
                on = True

            if on:
                # update current y-position selection
                if score > .3:
                    relative_y = relative_average_coordinate(heatmap, 0)
                    if root_relative_y is None:
                        root_relative_y = relative_y
                    acc_relative_y = accumulate(acc_relative_y, relative_y,
                                                accumulated_weight=1)

                # draw root and current y selection
                frame[int(root_relative_y * frame.shape[0]), :] = [0, 255, 0]
                frame[int(acc_relative_y * frame.shape[0]), :] = [0, 0, 255]

                # scroll
                self.mouse.scroll(0, int(
                    self.scrolling_sensitivity * (root_relative_y - acc_relative_y)
                ))
            cv2.imshow('Camera', text(cv2.flip(frame, 1), "Press ESC to quit"))
            cv2.imshow('Heatmap', cv2.resize(heatmap[:,::-1], frame.shape[1::-1]))
            if cv2.waitKey(1) & 0xFF == 27:  # esc to quit
                break

        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parser.parse_args()
    App(
        camera=args.camera,
        model_file=args.model_file,
        scrolling_sensitivity=args.sensitivity,
    ).run()
