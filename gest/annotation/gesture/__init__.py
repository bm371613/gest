from . import (
    static,
    closed_pinch,
    open_pinch,
)


def annotated_gesture_managers(data_path):
    return {
        "background": static.AnnotatedGestureManager(data_path / 'background'),
        "closed_pinch_left": closed_pinch.AnnotatedGestureManager(
            data_path / 'closed_pinch_left', hand="left"),
        "closed_pinch_right": closed_pinch.AnnotatedGestureManager(
            data_path / 'closed_pinch_right', hand="right"),
        "open_pinch_left": open_pinch.AnnotatedGestureManager(
            data_path / 'open_pinch_left', hand="left"),
        "open_pinch_right": open_pinch.AnnotatedGestureManager(
            data_path / 'open_pinch_right', hand="right"),
    }
