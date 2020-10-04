import random

import cv2

from gest.cv_gui import text

from . import static


def horizontal_line(frame, y: int):
    height, width, *_ = frame.shape
    return cv2.line(
        frame,
        (0, y),
        (width, y),
        (0, 0, 255),
        1,
    )


def vertical_line(frame, x: int):
    height, width, *_ = frame.shape
    return cv2.line(
        frame,
        (x, 0),
        (x, height),
        (0, 0, 255),
        1,
    )


def crosshead(frame, point):
    height, width, *_ = frame.shape
    frame = vertical_line(frame, int(point[0] * width))
    frame = horizontal_line(frame, int(point[1] * height))
    return frame


class AnnotatedGesture(static.AnnotatedGesture):
    def draw_annotations(self, resized_frame):
        frame = resized_frame
        point = self.annotations[0]['point']
        hand = self.annotations[0]['hand']
        frame = crosshead(frame, point)
        frame = text(frame, hand, point=point)
        return crosshead(frame, self.annotations[0]['point'])


class CapturingSession(static.CapturingSession):
    ANNOTATED_GESTURE_CLASS = AnnotatedGesture

    def draw_annotations(self, filipped_frame):
        frame = filipped_frame
        point = self.annotations[0]['point']
        hand = self.annotations[0]['hand']
        flipped_point = (1 - point[0], point[1])
        frame = crosshead(frame, flipped_point)
        frame = text(frame, hand, point=flipped_point)
        return frame


class SavedAnnotatedGesture(static.SavedAnnotatedGesture):
    ANNOTATED_GESTURE_CLASS = AnnotatedGesture


class AnnotatedGestureManager(static.AnnotatedGestureManager):
    CAPTURING_SESSION_CLASS = CapturingSession
    SAVED_ANNOTATED_GESTURE_CLASS = SavedAnnotatedGesture

    def __init__(self, data_path, hand=None):
        super().__init__(data_path)
        self.hand = hand

    def generate_annotations(self):
        return (
            {
                # FIXME: parameterize
                'point': [random.random() * .8 + .1, random.random() * .8 + .1],
                'hand': self.hand or random.choice(['left', 'right']),
            },
        )

