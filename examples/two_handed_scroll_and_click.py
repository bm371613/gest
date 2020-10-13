import argparse
import time

import cv2
import pynput.mouse

from gest.cv_gui import text, draw_inferred_crossheads, show_inference_result
from gest.inference import InferenceSession
from gest.math import relative_average_coordinate, accumulate

parser = argparse.ArgumentParser()
parser.add_argument("model_file", help="Model file")
parser.add_argument("--camera", help="Camera index", type=int, default=0)
parser.add_argument("--fps-limit", help="Frames per second limit", type=int)
parser.add_argument("--sensitivity", help="Scrolling sensitivity", type=int, default=20)


class App:

    def __init__(self, camera, model_file, fps_limit, scrolling_sensitivity):
        self.camera = camera
        self.inference_session = InferenceSession(model_file)
        self.mouse = pynput.mouse.Controller()

        self.fps_limit = fps_limit
        self.scrolling_sensitivity = scrolling_sensitivity
        self.score_threshold = .5

    def run(self):
        capture = cv2.VideoCapture(self.camera)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        button_down = None
        button_down_since = None
        scroll_acc = 0
        scroll_distance = 0
        last_time = None
        last_click = None
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            this_time = time.time()

            # scrolling before fps limiting for smooth motion
            if last_time:
                scroll_distance += scroll_acc * (last_time - this_time) * self.scrolling_sensitivity
                self.mouse.scroll(0, int(scroll_distance))
                scroll_distance -= int(scroll_distance)

            # fps limiting
            if self.fps_limit and last_time and (this_time - last_time) * self.fps_limit < 1:
                continue

            # inference
            inference_result = self.inference_session.cv2_run(frame)
            left, right = inference_result

            # actions
            left_x, left_y = relative_average_coordinate(left, (1, 0))
            right_x, right_y = relative_average_coordinate(right, (1, 0))
            button_down_now = None
            scroll_now = 0

            if left.max() < self.score_threshold or right.max() < self.score_threshold:
                pass
            elif left_x < right_x:
                button_down_now = 'double click'
            elif abs(left_y - right_y) < .05:
                if left_x - right_x < .1:
                    button_down_now = 'click'
                else:
                    button_down_now = 'right click'
            elif abs(left_y - right_y) > .1:
                scroll_now = right_y - left_y
            if button_down != button_down_now:
                button_down_since = this_time
                if button_down_now == 'click' and (last_click is None or this_time - last_click > .5):
                    self.mouse.click(pynput.mouse.Button.left)
                    last_click = this_time
                if button_down_now == 'double click' and (last_click is None or this_time - last_click > .5):
                    self.mouse.click(pynput.mouse.Button.left, 2)
                    last_click = this_time
                button_down = button_down_now
            elif button_down_now == 'right click' and .5 < this_time - button_down_since and \
                    (last_click is None or last_click < button_down_since):
                self.mouse.click(pynput.mouse.Button.right)
                last_click = this_time
            scroll_acc = accumulate(scroll_acc, scroll_now)

            frame = draw_inferred_crossheads(frame, inference_result)
            cv2.imshow('Camera', text(cv2.flip(frame, 1), "Press ESC to quit"))
            show_inference_result(frame, inference_result)

            if cv2.waitKey(1) & 0xFF == 27:  # esc to quit
                break
            last_time = this_time

        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parser.parse_args()
    App(
        camera=args.camera,
        model_file=args.model_file,
        fps_limit=args.fps_limit,
        scrolling_sensitivity=args.sensitivity,
    ).run()
