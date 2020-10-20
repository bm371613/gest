import argparse
import threading
import time

import cv2
import pynput.mouse

from gest.cv_gui import text, draw_inferred_crossheads, show_inference_result
from gest.inference import InferenceSession
from gest.math import relative_average_coordinate

parser = argparse.ArgumentParser()
parser.add_argument("--camera", help="Camera index", type=int, default=0)
parser.add_argument("--model", help="Model file")
parser.add_argument("--sensitivity", help="Scrolling sensitivity", type=int, default=20)


class App:

    def __init__(self, camera, model_file, scrolling_sensitivity):
        self.camera = camera
        self.inference_session = InferenceSession(model_file)
        self.mouse = pynput.mouse.Controller()

        self.scrolling_sensitivity = scrolling_sensitivity
        self.score_threshold = .5
        self.scrolling_speed = 0

    def scroll_forever(self):
        last_time = time.time()
        distance = 0
        while True:
            time.sleep(0.1)
            now = time.time()
            distance += (now - last_time) * self.scrolling_speed * self.scrolling_sensitivity
            self.mouse.scroll(0, int(distance))
            distance -= int(distance)
            last_time = now

    def run(self):
        scrolling_thread = threading.Thread(target=self.scroll_forever)
        scrolling_thread.daemon = True
        scrolling_thread.start()

        capture = cv2.VideoCapture(self.camera)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        button_down = None
        button_down_since = None
        last_click = None
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            now = time.time()

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
                scroll_now = left_y - right_y

            if button_down != button_down_now:
                button_down_since = now
                if button_down_now == 'click' and (last_click is None or now - last_click > .5):
                    self.mouse.click(pynput.mouse.Button.left)
                    last_click = now
                if button_down_now == 'double click' and (last_click is None or now - last_click > .5):
                    self.mouse.click(pynput.mouse.Button.left, 2)
                    last_click = now
                button_down = button_down_now
            elif button_down_now == 'right click' and .5 < now - button_down_since and \
                    (last_click is None or last_click < button_down_since):
                self.mouse.click(pynput.mouse.Button.right)
                last_click = now

            self.scrolling_speed = scroll_now

            frame = draw_inferred_crossheads(frame, inference_result)
            cv2.imshow('Camera', text(cv2.flip(frame, 1), "Press ESC to quit"))
            show_inference_result(frame, inference_result)

            if cv2.waitKey(1) & 0xFF == 27:  # esc to quit
                break

        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parser.parse_args()
    App(
        camera=args.camera,
        model_file=args.model,
        scrolling_sensitivity=args.sensitivity,
    ).run()
