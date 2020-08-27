import argparse
import collections
import time

import cv2

from gest import model
from gest.cv_gui import text


parser = argparse.ArgumentParser()
parser.add_argument("model_path", help="Model path")
parser.add_argument("--camera", help="Camera index", type=int, default=0)
parser.add_argument("--history", help="History size", type=int, default=10)


HistoryEntry = collections.namedtuple('HistoryEntry', 'at score')


class App:

    def __init__(self, camera, model, history_size):
        self.camera = camera
        self.model = model
        self.history_size = history_size

    def run(self):
        video_capture = cv2.VideoCapture(self.camera)
        history = collections.deque()
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            now = time.time()
            score = model.score(self.model, frame)
            history.append(HistoryEntry(now, score))
            display = cv2.flip(frame, 1)
            if len(history) == self.history_size:
                fps = (len(history) - 1) / (history[-1].at - history[0].at)
                avg_score = sum(e.score for e in history) / len(history)
                display = text(display, f'fps {fps:.0f}, score {avg_score:.0%}, avg over {len(history)}')
                history.popleft()
            cv2.imshow('Demo', display)
            if cv2.waitKey(1) & 0xFF == 27:  # esc to quit
                break

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parser.parse_args()
    App(
        camera=args.camera,
        model=model.load(args.model_path),
        history_size=args.history,
    ).run()
