import argparse
import time

import cv2

from gest.cv_gui import show_inference_result, text, draw_inferred_crossheads
from gest.inference import InferenceSession
from gest.math import accumulate

parser = argparse.ArgumentParser()
parser.add_argument("--camera", help="Camera index", type=int, default=0)
parser.add_argument("--model", help="Model file")


class App:

    def __init__(self, camera, model_file):
        self.camera = camera
        self.inference_session = InferenceSession(model_file)

    def run(self):
        capture = cv2.VideoCapture(self.camera)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        last_time = time.time()
        fps = None
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            result = self.inference_session.cv2_run(frame)

            now = time.time()
            fps = accumulate(fps, 1 / (now - last_time))
            last_time = now

            frame = draw_inferred_crossheads(frame, result)
            show_inference_result(frame, result)
            cv2.imshow('Camera', text(cv2.flip(frame, 1), f"{fps:.1f} fps | Press ESC to quit"))
            if cv2.waitKey(1) & 0xFF == 27:  # esc to quit
                break
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parser.parse_args()
    App(
        camera=args.camera,
        model_file=args.model,
    ).run()
