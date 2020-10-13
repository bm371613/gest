import random

from gest.cv_gui import text, crosshead

from . import static


class AnnotatedGesture(static.AnnotatedGesture):
    def draw_annotations(self, resized_frame):
        frame = resized_frame
        point = self.annotations[0]['point']
        hand = self.annotations[0]['hand']
        frame = crosshead(frame, point)
        frame = text(frame, hand, point=point)
        return crosshead(frame, self.annotations[0]['point'])


class CapturingSession(static.CapturingSession):

    def draw_annotations(self, filipped_frame):
        frame = filipped_frame
        point = self.annotations[0]['point']
        hand = self.annotations[0]['hand']
        flipped_point = (1 - point[0], point[1])
        frame = crosshead(frame, flipped_point)
        frame = text(frame, hand, point=flipped_point)
        return frame


class AnnotatedGestureManager(static.AnnotatedGestureManager):

    def __init__(self, data_path, hand=None):
        super().__init__(
            data_path,
            annotated_gesture_class=AnnotatedGesture,
            capturing_session_class=CapturingSession,
        )
        self.hand = hand

    def generate_annotations(self):
        return (
            {
                # FIXME: parameterize
                'point': [random.random() * .8 + .1, random.random() * .8 + .1],
                'hand': self.hand or random.choice(['left', 'right']),
            },
        )

