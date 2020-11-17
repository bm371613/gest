import cv2
import numpy as np

from gest.math import relative_average_coordinate

LEFT_COLOR = np.array((0., 0., 1.))
RIGHT_COLOR = np.array((0., 1., 0.))
OPEN_COLOR = np.array((1., 0., 0.))


def text(frame, text, scale=1, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, point=(0, 1),
         fg=(255, 255, 255), bg=(0, 0, 0)):
    x, y = int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0] - thickness * 10)
    frame = cv2.putText(frame, text, (x - 1, y), font, scale, bg, thickness)
    frame = cv2.putText(frame, text, (x + 1, y), font, scale, bg, thickness)
    frame = cv2.putText(frame, text, (x, y - 1), font, scale, bg, thickness)
    frame = cv2.putText(frame, text, (x, y + 1), font, scale, bg, thickness)
    frame = cv2.putText(frame, text, (x, y), font, scale, fg, thickness)
    return frame


def horizontal_line(frame, y: int, color=(0, 0, 255)):
    height, width, *_ = frame.shape
    return cv2.line(
        frame,
        (0, y),
        (width, y),
        color,
        1,
    )


def vertical_line(frame, x: int, color=(0, 0, 255)):
    height, width, *_ = frame.shape
    return cv2.line(
        frame,
        (x, 0),
        (x, height),
        color,
        1,
    )


def crosshead(frame, x, y, color=(0, 0, 255)):
    height, width, *_ = frame.shape
    frame = vertical_line(frame, int(x * width), color)
    frame = horizontal_line(frame, int(y * height), color)
    return frame


def show_inference_result(frame, inference_result):
    (left, open_left), (right, open_right) = inference_result
    display = np.zeros((*left.shape, 3))
    np.maximum(display, np.tensordot(left, LEFT_COLOR, axes=0), out=display)
    np.maximum(display, np.tensordot(right, RIGHT_COLOR, axes=0), out=display)
    np.maximum(display, np.tensordot(np.maximum(open_left * left, open_right * right), OPEN_COLOR, axes=0), out=display)
    display = cv2.flip(cv2.resize(display, frame.shape[1::-1]), 1)
    display = text(display, f'Left: {left.max():.0%}', point=(0, 1))
    if left.max() > .5:
        color = np.maximum(LEFT_COLOR, np.multiply(OPEN_COLOR, open_left[left > .5].mean()))
        display = text(display, f'Open: {open_left[left > .5].mean():.0%}', fg=color, point=(0, .8))
    display = text(display, f'Right: {right.max():.0%}', point=(.5, 1))
    if right.max() > .5:
        color = np.maximum(RIGHT_COLOR, np.multiply(OPEN_COLOR, open_right[right > .5].mean()))
        display = text(display, f'Open: {open_right[right > .5].mean():.0%}', fg=color, point=(0.5, .8))
    cv2.imshow('Heatmap', display)


def draw_inferred_crossheads(frame, inference_result):
    (left, open_left), (right, open_right) = inference_result
    if left.max() > .5:
        frame = crosshead(
            frame,
            *relative_average_coordinate(left, (1, 0)),
            color=np.maximum(
                LEFT_COLOR,
                np.multiply(OPEN_COLOR, open_left[left > .5].mean()),
            ) * 255,
        )
    if right.max() > .5:
        frame = crosshead(
            frame,
            *relative_average_coordinate(right, (1, 0)),
            color=np.maximum(
                RIGHT_COLOR,
                np.multiply(OPEN_COLOR, open_right[right > .5].mean()),
            ) * 255,
        )
    return frame
