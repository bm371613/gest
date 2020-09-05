from . import (
    base,
    static,
    y_selection,
)


def annotated_gesture_managers(data_path):
    return [
        static.AnnotatedGestureManager('no gesture', data_path / '0'),
        static.AnnotatedGestureManager('open hand', data_path / '1'),
        y_selection.AnnotatedGestureManager('pinch', data_path / '2'),
    ]
