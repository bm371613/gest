import random

from gest.cv_gui import text, crosshead

from . import static


class AnnotatedGesture(static.AnnotatedGesture):
    def draw_annotations(self, resized_frame):
        frame = resized_frame
        for annotation in self.annotations:
            x = annotation['x']
            y = annotation['y']
            frame = crosshead(frame, x, y)
            if annotation['label'] == 'open_pinch_top':
                hand = annotation['hand']
                frame = text(frame, hand, point=(x, y))
        return frame


class CapturingSession(static.CapturingSession):
    def draw_annotations(self, filipped_frame):
        frame = filipped_frame
        for annotation in self.annotations:
            x = annotation['x']
            y = annotation['y']
            flipped_point = (1 - x, y)
            frame = crosshead(frame, *flipped_point)
            if annotation['label'] == 'open_pinch_top':
                hand = annotation['hand']
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
        x = random.random() * 0.8 + .1
        y_span = random.random() * 0.05 + 0.025
        y_center = random.random() * (0.8 - y_span) + .1 + y_span / 2
        return (
            {
                "label": "open_pinch_bottom",
                'hand': self.hand,
                'x': x,
                'y': y_center + y_span / 2,
            },
            {
                "label": "open_pinch_top",
                'hand': self.hand,
                'x': x,
                'y': y_center - y_span / 2,
            },
        )

