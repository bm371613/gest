import json
import random
import typing

import cv2

from gest.cv_gui import text

from . import base


def horizontal_line(frame, y: float):
    height, width, *_ = frame.shape
    return cv2.line(
        frame,
        (0, int(height * y)),
        (width, int(height * y)),
        (0, 0, 255),
        1,
    )


class AnnotatedGesture(base.AnnotatedGesture, base.PlaybackSession):

    def __init__(self, label, y: float, frame, name):
        self.label = label
        self.y = y
        self.frame = frame
        self.name = name

    def start_playback_session(self, at):
        return self

    def render(self, at, size=None):
        frame = self.frame
        if size is not None:
            frame = cv2.resize(frame, size)
        return text(horizontal_line(frame, self.y), self.label)


class CapturingSession(base.CapturingSession):

    def __init__(self, started_at, countdown, label, y):
        self.started_at = started_at
        self.countdown = countdown
        self.label = label
        self.y = y
        self._result = None

    def process(self, at, frame):
        display = cv2.flip(frame, 1)
        if self.result():
            return display
        elif at - self.started_at < self.countdown:
            return text(
                horizontal_line(display, self.y),
                f'[{self.label}] in {int(self.started_at + self.countdown - at)}s',
            )
        else:
            self._result = AnnotatedGesture(self.label, self.y, frame, name=str(int(at)))
            return display

    def result(self) -> typing.Optional[AnnotatedGesture]:
        return self._result


class SavedAnnotatedGesture(base.SavedAnnotatedGesture):

    def __init__(self, path, label):
        self.path = path
        self.label = label

    def load(self) -> AnnotatedGesture:
        return AnnotatedGesture(
            label=self.label,
            y=json.load(self.path.with_suffix('.json').open()),
            frame=cv2.imread(str(self.path)),
            name=self.path.stem,
        )

    def remove(self):
        self.path.unlink()
        self.path.with_suffix('.json').unlink()


class AnnotatedGestureManager(base.AnnotatedGestureManager):

    def __init__(self, label, data_path):
        self.label = label
        self.data_path = data_path

    def start_capturing_session(self, at, *, countdown=0) -> CapturingSession:
        return CapturingSession(
            started_at=at,
            countdown=max(countdown, 3),
            label=self.label,
            y=random.random() * .9 + .05,
        )

    def save(self, annotated_gesture: AnnotatedGesture) -> SavedAnnotatedGesture:
        path = self.data_path / f'{annotated_gesture.name}.jpg'
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), annotated_gesture.frame)
        json.dump(annotated_gesture.y, path.with_suffix('.json').open('w'))
        return SavedAnnotatedGesture(path, self.label)
