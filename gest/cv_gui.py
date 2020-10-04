import cv2


def text(frame, text, scale=1, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, point=(0, 1)):
    x, y = int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0] - thickness * 10)
    fg = (255, 255, 255)
    bg = (0, 0, 0)
    frame = cv2.putText(frame, text, (x - 1, y), font, scale, bg, thickness)
    frame = cv2.putText(frame, text, (x + 1, y), font, scale, bg, thickness)
    frame = cv2.putText(frame, text, (x, y - 1), font, scale, bg, thickness)
    frame = cv2.putText(frame, text, (x, y + 1), font, scale, bg, thickness)
    frame = cv2.putText(frame, text, (x, y), font, scale, fg, thickness)
    return frame
