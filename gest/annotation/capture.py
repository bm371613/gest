import argparse
import pathlib
import time

import cv2
import numpy as np

from gest.annotation.gesture import annotated_gesture_managers
from gest.cv_gui import text

parser = argparse.ArgumentParser()
parser.add_argument("data_path", help="Data path")
parser.add_argument("--camera", help="Camera index", type=int, default=0)
parser.add_argument("--countdown", help="Countdown interval", type=int, default=3)
parser.add_argument('--order', nargs='+', help='Auto mode order', default=(
    'closed_pinch_left', 'closed_pinch_right', 'background',
))


class App:

    def __init__(self, camera, data_path, countdown, order):
        self.camera = camera
        self.countdown = countdown
        self.data_path = data_path
        self.order = order

        self.annotated_gesture_managers = annotated_gesture_managers(data_path)

        self.history = []
        self.order_ix = 0
        self.gesture_name = None
        self.auto = False
        self.capturing_session = None
        self.playback_session = None

    def run(self):
        self.data_path.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(self.camera)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            now = time.time()
            self.handle_frame(now, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # esc to quit
                break
            else:
                self.handle_key(now, key)

            self.finalize_capturing_maybe(now)

        capture.release()
        cv2.destroyAllWindows()

    def handle_frame(self, at, frame):
        display = cv2.vconcat([
            text(
                np.zeros_like(frame)[:50],
                f'{self.gesture_name}: {self.capturing_session.message(at)}'
                if self.capturing_session
                else 'ESC: quit; a: start/stop; d: delete',
            ),
            self.capturing_session.process(at, frame) if self.capturing_session else cv2.flip(frame, 1),
            self.playback_session.render(at, frame.shape[:-1][::-1]) if self.playback_session else text(
                np.zeros_like(frame),
                'Last labeled image will be here',
            ),
        ])
        cv2.imshow('Capture and Label', display)

    def handle_key(self, at, key):
        if key == ord('d'):
            if self.history:
                self.history.pop(-1).remove()
                self.playback_session = None
            if self.history:
                annotated = self.history[-1].load()
                self.playback_session = annotated.start_playback_session(at)
            else:
                self.playback_session = None
        elif key == ord('a'):
            if self.auto:
                self.auto = False
                self.capturing_session = None
            else:
                self.auto = True
                self.start_capturing(at)

    def start_capturing(self, at, gesture_name=None, countdown=None):
        if gesture_name is None:
            gesture_name = self.order[self.order_ix % len(self.order)]
            self.order_ix += 1
        self.gesture_name = gesture_name
        manager = self.annotated_gesture_managers[self.gesture_name]
        self.capturing_session = manager.start_capturing_session(
            at,
            countdown=self.countdown if countdown is None else countdown,
        )

    def finalize_capturing_maybe(self, at):
        if self.capturing_session is None:
            return
        annotated = self.capturing_session.result()
        if annotated is None:
            return
        self.history.append(
            self.annotated_gesture_managers[self.gesture_name].save(annotated)
        )
        self.playback_session = annotated.start_playback_session(at)
        if self.auto:
            self.start_capturing(at)
        else:
            self.capturing_session = None


if __name__ == '__main__':
    args = parser.parse_args()
    App(
        camera=args.camera,
        data_path=pathlib.Path(args.data_path),
        countdown=args.countdown,
        order=args.order,
    ).run()
