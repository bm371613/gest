import argparse

import cv2

from gest.cv_gui import show_inference_result, text, draw_inferred_crossheads
from gest.inference import CvCameraInferencePipeline
from gest.math import accumulate

parser = argparse.ArgumentParser()
parser.add_argument("--camera", help="Camera index", type=int, default=0)
parser.add_argument("--model", help="Model file")


class App:

    def __init__(self, camera, model_file):
        self.pipeline = CvCameraInferencePipeline(camera, model_file)

    def run(self):
        fps = None
        latency = None
        with self.pipeline.threaded() as stream:
            for item in stream:
                fps = accumulate(fps, item.fps)
                latency = accumulate(latency, item.latency)
                frame = item.frame
                frame = draw_inferred_crossheads(frame, item.inference_result)
                frame = cv2.flip(frame, 1)
                frame = text(frame, f"fps {fps: 2.0f}", point=(0, .5))
                frame = text(frame, f"latency {latency:.2f}s", point=(0, .75))
                frame = text(frame, "Press ESC to quit")
                cv2.imshow('Camera', frame)
                show_inference_result(frame, item.inference_result)
                if cv2.waitKey(1) & 0xFF == 27:  # esc to quit
                    break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parser.parse_args()
    App(
        camera=args.camera,
        model_file=args.model,
    ).run()
