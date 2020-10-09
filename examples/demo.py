import argparse

import cv2
import numpy as np

from gest.cv_gui import text
from gest.inference import InferenceSession

parser = argparse.ArgumentParser()
parser.add_argument("model_file", help="Model file")
parser.add_argument("--camera", help="Camera index", type=int, default=0)


class App:

    def __init__(self, camera, model_file):
        self.camera = camera
        self.inference_session = InferenceSession(model_file)

    def run(self):
        capture = cv2.VideoCapture(self.camera)
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            cv2.imshow('Camera', text(cv2.flip(frame, 1), "Press ESC to quit"))
            left, right = self.inference_session.cv2_run(frame)
            merged = np.zeros((*left.shape, 3))
            merged[:, :, 2] = left  # red for left hand
            merged[:, :, 1] = right  # green for right hand
            cv2.imshow('Heatmap', cv2.flip(cv2.resize(merged, frame.shape[1::-1]), 1))
            if cv2.waitKey(1) & 0xFF == 27:  # esc to quit
                break
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parser.parse_args()
    App(
        camera=args.camera,
        model_file=args.model_file,
    ).run()
