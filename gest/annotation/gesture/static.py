import json
import typing

import cv2

from . import base


class AnnotatedGesture(base.AnnotatedGesture, base.PlaybackSession):

    def __init__(self, name, frame, annotations=()):
        self.name = name
        self.frame = frame
        self.annotations = annotations

    def start_playback_session(self, at):
        return self

    def draw_annotations(self, resized_frame):
        return resized_frame

    def render(self, at, size=None):
        frame = self.frame
        if size is not None:
            frame = cv2.resize(frame, size)
        return self.draw_annotations(frame)


class CapturingSession(base.CapturingSession):
    ANNOTATED_GESTURE_CLASS = AnnotatedGesture

    def __init__(self, started_at, countdown, annotations=()):
        self.started_at = started_at
        self.countdown = countdown
        self.annotations = annotations
        self._result = None

    def message(self, at):
        return f'capturing in {int(self.countdown + self.started_at - at)}s'

    def draw_annotations(self, filipped_frame):
        return filipped_frame

    def process(self, at, frame):
        if self.result():
            return cv2.flip(frame, 1)
        elif at - self.started_at < self.countdown:
            return self.draw_annotations(filipped_frame=cv2.flip(frame, 1))
        else:
            self._result = self.ANNOTATED_GESTURE_CLASS(name=str(int(at)), frame=frame, annotations=self.annotations)
            return cv2.flip(frame, 1)

    def result(self) -> typing.Optional[AnnotatedGesture]:
        return self._result


class SavedAnnotatedGesture(base.SavedAnnotatedGesture):

    ANNOTATED_GESTURE_CLASS = AnnotatedGesture

    def __init__(self, path):
        self.path = path

    @property
    def annotations_path(self):
        return self.path.with_suffix('.json')

    @classmethod
    def save(cls, annotated_gesture, path):
        result = cls(path=path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), annotated_gesture.frame)
        json.dump(annotated_gesture.annotations, result.annotations_path.open('w'))
        return result

    def load(self) -> AnnotatedGesture:
        return self.ANNOTATED_GESTURE_CLASS(
            name=self.path.stem,
            frame=cv2.imread(str(self.path)),
            annotations=(
                json.load(self.annotations_path.open())
                if self.annotations_path.exists()
                else ()
            ),
        )

    def remove(self):
        self.path.unlink()
        self.annotations_path.unlink()


class AnnotatedGestureManager(base.AnnotatedGestureManager):
    CAPTURING_SESSION_CLASS = CapturingSession
    SAVED_ANNOTATED_GESTURE_CLASS = SavedAnnotatedGesture

    def __init__(self, data_path):
        self.data_path = data_path

    def generate_annotations(self):
        return ()

    def start_capturing_session(self, at, *, countdown=0) -> CapturingSession:
        return self.CAPTURING_SESSION_CLASS(
            started_at=at,
            countdown=countdown,
            annotations=self.generate_annotations(),
        )

    def save(self, annotated_gesture: AnnotatedGesture) -> SavedAnnotatedGesture:
        path = self.data_path / f'{annotated_gesture.name}.jpg'
        return self.SAVED_ANNOTATED_GESTURE_CLASS.save(annotated_gesture, path)

    def saved(self) -> typing.Iterable[SavedAnnotatedGesture]:
        for path in sorted(self.data_path.glob('*.jpg')):
            yield self.SAVED_ANNOTATED_GESTURE_CLASS(path)
