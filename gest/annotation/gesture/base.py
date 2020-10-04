import typing


class PlaybackSession:

    def render(self, at, size=None):
        raise NotImplementedError()


class AnnotatedGesture:

    def start_playback_session(self, at) -> PlaybackSession:
        raise NotImplementedError()


class CapturingSession:

    def message(self, at):
        return ''

    def process(self, at, frame):
        raise NotImplementedError()

    def result(self) -> typing.Optional[AnnotatedGesture]:
        raise NotImplementedError()


class SavedAnnotatedGesture:

    def load(self) -> AnnotatedGesture:
        raise NotImplementedError()

    def remove(self):
        raise NotImplementedError()


class AnnotatedGestureManager:

    def start_capturing_session(self, at, *, countdown=0) -> CapturingSession:
        raise NotImplementedError()

    def save(self, annotated_gesture: AnnotatedGesture) -> SavedAnnotatedGesture:
        raise NotImplementedError()

    def saved(self) -> typing.Iterable[SavedAnnotatedGesture]:
        raise NotImplementedError()
