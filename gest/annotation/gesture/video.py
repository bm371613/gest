import typing

import cv2

from . import base


class PlaybackSession(base.PlaybackSession):

    def __init__(self, frames, fps, started_at):
        self.frames = frames
        self.fps = fps
        self.started_at = started_at

    def render(self, at, size=None):
        frame = self.frames[int((at - self.started_at) * self.fps) % len(self.frames)]
        if size is not None:
            frame = cv2.resize(frame, size)
        return frame


class AnnotatedGesture(base.AnnotatedGesture):

    def __init__(self, name, frames, fps):
        self.name = name
        self.frames = frames
        self.fps = fps

    def start_playback_session(self, at):
        return PlaybackSession(self.frames, self.fps, at)


class CapturingSession(base.CapturingSession):

    def __init__(self, started_at, countdown, duration, annotated_gesture_class=AnnotatedGesture):
        self.started_at = started_at
        self.countdown = countdown
        self.duration = duration
        self.annotated_gesture_class = annotated_gesture_class
        self._frames = []
        self._result = None

    def message(self, at):
        if at - self.started_at < self.countdown:
            return f'capturing in {int(self.countdown + self.started_at - at)}s'
        else:
            return f'{int(self.duration + self.countdown + self.started_at - at)}s left'

    def process(self, at, frame):
        if self.countdown < at - self.started_at < self.countdown + self.duration:
            self._frames.append(frame)
        if self._result is None and at - self.started_at >= self.countdown + self.duration:
            self._result = self.annotated_gesture_class(
                name=str(int(at)),
                frames=self._frames,
                fps=len(self._frames) / self.duration,
            )
        return cv2.flip(frame, 1)

    def result(self) -> typing.Optional[AnnotatedGesture]:
        return self._result


class SavedAnnotatedGesture(base.SavedAnnotatedGesture):

    def __init__(self, path, annotated_gesture_class=AnnotatedGesture):
        self.path = path
        self.annotated_gesture_class = annotated_gesture_class

    @classmethod
    def save(cls, annotated_gesture, path, annotated_gesture_class):
        result = cls(path=path, annotated_gesture_class=annotated_gesture_class)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            annotated_gesture.fps,
            tuple(reversed(annotated_gesture.frames[0].shape[:2])),
        )
        for frame in annotated_gesture.frames:
            writer.write(frame)
        writer.release()
        return result

    def load(self) -> AnnotatedGesture:
        frames = []
        capture = cv2.VideoCapture(str(self.path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            frames.append(frame)
        capture.release()
        return self.annotated_gesture_class(
            name=self.path.stem,
            frames=frames,
            fps=fps,
        )

    def remove(self):
        self.path.unlink()


class AnnotatedGestureManager(base.AnnotatedGestureManager):

    def __init__(self, data_path, capturing_session_class=CapturingSession,
                 annotated_gesture_class=AnnotatedGesture,
                 saved_annotated_gesture_class=SavedAnnotatedGesture):
        self.data_path = data_path
        self.capturing_session_class = capturing_session_class
        self.annotated_gesture_class = annotated_gesture_class
        self.saved_annotated_gesture_class = saved_annotated_gesture_class

    def start_capturing_session(self, at, *, countdown=0) -> CapturingSession:
        return self.capturing_session_class(
            started_at=at,
            countdown=countdown,
            annotated_gesture_class=self.annotated_gesture_class,
            duration=10,
        )

    def save(self, annotated_gesture: AnnotatedGesture) -> SavedAnnotatedGesture:
        path = self.data_path / f'{annotated_gesture.name}.mp4'
        return self.saved_annotated_gesture_class.save(
            annotated_gesture, path, self.annotated_gesture_class,
        )

    def saved(self) -> typing.Iterable[SavedAnnotatedGesture]:
        for path in sorted(self.data_path.glob('*.mp4')):
            yield self.saved_annotated_gesture_class(
                path, annotated_gesture_class=self.annotated_gesture_class,
            )
