import random

from gest.cv_gui import text, crosshead

from . import static


class AnnotatedGesture(static.AnnotatedGesture):
    def draw_annotations(self, resized_frame):
        frame = resized_frame
        x = self.annotations[0]['x']
        y = self.annotations[0]['y']
        hand = self.annotations[0]['hand']
        frame = crosshead(frame, x, y)
        frame = text(frame, hand, point=(x, y))
        return frame


class CapturingSession(static.CapturingSession):

    def draw_annotations(self, filipped_frame):
        frame = filipped_frame
        x = self.annotations[0]['x']
        y = self.annotations[0]['y']
        hand = self.annotations[0]['hand']
        flipped_point = (1 - x, y)
        frame = crosshead(frame, *flipped_point)
        frame = text(frame, hand, point=flipped_point)
        return frame


class AnnotatedGestureManager(static.AnnotatedGestureManager):

    def __init__(self, data_path, hand):
        super().__init__(
            data_path,
            annotated_gesture_class=AnnotatedGesture,
            capturing_session_class=CapturingSession,
        )
        self.hand = hand

    def generate_annotations(self):
        return (
            {
                "label": "closed_pinch",
                'hand': self.hand,
                'x': random.random() * .8 + .1,
                'y': random.random() * .8 + .1,
            },
        )

