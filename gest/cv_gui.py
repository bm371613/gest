import cv2
import numpy as np

from gest.math import relative_average_coordinate

LEFT_COLOR = (0, 0, 255)
RIGHT_COLOR = (0, 255, 0)
LEFT_COLOR_IX = 2
RIGHT_COLOR_IX = 1


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
    left, right = inference_result
    display = np.zeros((*left.shape, 3))
    display[:, :, LEFT_COLOR_IX] = left  # red for left hand
    display[:, :, RIGHT_COLOR_IX] = right  # green for right hand
    display = cv2.flip(cv2.resize(display, frame.shape[1::-1]), 1)
    display = text(display, f'Left: {left.max():.0%}', fg=LEFT_COLOR)
    display = text(display, f'Right: {right.max():.0%}', fg=RIGHT_COLOR, point=(.5, 1))
    cv2.imshow('Heatmap', display)


def draw_inferred_crossheads(frame, inference_result):
    left, right = inference_result
    if left.max() > .5:
        frame = crosshead(
            frame,
            *relative_average_coordinate(left, (1, 0)),
            color=LEFT_COLOR,
        )
    if right.max() > .5:
        frame = crosshead(
            frame,
            *relative_average_coordinate(right, (1, 0)),
            color=RIGHT_COLOR,
        )
    return frame
