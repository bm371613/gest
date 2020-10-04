import argparse
import pathlib
import time

import cv2

from gest.annotation.gesture import annotated_gesture_managers

parser = argparse.ArgumentParser()
parser.add_argument("data_path", help="Data path")
parser.add_argument("gesture_name", help="Gesture name")
parser.add_argument("--ix", help="Annotation index", type=int, default=1)
parser.add_argument("--time", help="Playback time", type=float, default=1)


class App:

    def __init__(self, data_path, gesture_name, ix, playback_time):
        self.items = list(annotated_gesture_managers(data_path)[gesture_name].saved())
        self.ix = ix
        self.playback_time = playback_time

        self.auto = False
        self.playback_session = None

    def run(self):
        started_at = None
        window_name = 'review'
        while self.ix < len(self.items):
            now = time.time()
            if self.playback_session is None:
                started_at = now
                self.playback_session = self.items[self.ix].load().start_playback_session(now)
            display = self.playback_session.render(now)
            cv2.imshow(window_name, display)
            cv2.setWindowTitle(window_name, f'Review {self.ix + 1}/{len(self.items)}')

            key = cv2.waitKey(10 or (self.auto and now > started_at + self.playback_session)) & 0xFF
            if key == 27:  # esc to quit
                break
            elif key == ord('d'):
                self.items[self.ix].remove()
                del self.items[self.ix]
                self.playback_session = None
            elif key == ord('a'):
                self.auto = not self.auto
            elif key == ord('p'):
                self.ix -= 1
                self.playback_session = None
            elif key == ord('n') or (self.auto and now > started_at + self.playback_time):
                self.ix += 1
                self.playback_session = None
        cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parser.parse_args()
    App(
        data_path=pathlib.Path(args.data_path),
        gesture_name=args.gesture_name,
        ix=args.ix - 1,
        playback_time=args.time,
    ).run()
