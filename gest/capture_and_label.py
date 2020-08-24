import argparse
import collections
import pathlib
import time

import cv2
import numpy as np

from gest.cv_gui import text

LabeledImage = collections.namedtuple('LabeledImage', 'frame label path')
Countdown = collections.namedtuple('Countdown', 'at ix')

parser = argparse.ArgumentParser()
parser.add_argument("data_path", help="Data path")
parser.add_argument("--camera", help="Camera index", type=int, default=0)
parser.add_argument("--countdown", help="Countdown interval", type=int, default=3)


class App:

    def __init__(self, camera, data_path, countdown_interval):
        self.camera = camera
        self.data_path = data_path
        self.countdown_interval = countdown_interval

    def label_from_index(self, ix):
        return '01'[ix % 2]

    def save_label(self, frame, label, name):
        path = self.data_path / label / f'{name}.jpg'
        print(f'labeling {path} as {label}')
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame)
        return LabeledImage(frame, label, path)

    def next_countdown(self, now, last=None):
        return Countdown(
            now + self.countdown_interval,
            last.ix + 1 if last is not None else 0,
        )

    def run(self):
        history = []
        video_capture = cv2.VideoCapture(self.camera)
        countdown = None
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            now = time.time()
            display = np.array(frame[:, ::-1, :])
            if countdown:
                display = text(
                    display,
                    f'Labeling as {self.label_from_index(countdown.ix)}'
                    f' in {int(countdown.at - now)}s',
                )
            display = cv2.vconcat([
                display,
                text(
                    history[-1].frame, f'Label: {history[-1].label}',
                ) if history else text(
                    np.zeros_like(display),
                    'Last labeled image will be here',
                ),
            ])
            cv2.imshow('Capture and Label', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif ord('0') <= key <= ord('9'):
                history.append(self.save_label(
                    frame,
                    self.label_from_index(key - ord('0')),
                    str(int(now)),
                ))
            elif key == ord('d'):
                if history:
                    path = history.pop(-1).path
                    print(f'removing {path}')
                    path.unlink()
            elif key == ord('p'):
                countdown = self.next_countdown(now) if countdown is None else None
            elif countdown is not None and now > countdown.at:
                history.append(self.save_label(
                    frame,
                    self.label_from_index(countdown.ix),
                    str(int(now)),
                ))
                countdown = self.next_countdown(now, countdown)

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parser.parse_args()
    App(
        camera=args.camera,
        data_path=pathlib.Path(args.data_path),
        countdown_interval=args.countdown,
    ).run()
