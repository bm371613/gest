from . import (
    static,
    single_landmark,
)


def annotated_gesture_managers(data_path):
    return {
        "background": static.AnnotatedGestureManager(data_path / 'background'),
        "pinch_left": single_landmark.AnnotatedGestureManager(
            data_path / 'pinch_left', hand="left"),
        "pinch_right": single_landmark.AnnotatedGestureManager(
            data_path / 'pinch_right', hand="right"),
    }
