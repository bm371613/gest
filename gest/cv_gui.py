import cv2


def text(frame, text, scale=1, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    x, y = 0, frame.shape[0] - thickness * 10
    fg = (255, 255, 255)
    bg = (0, 0, 0)
    frame = cv2.putText(frame, text, (x - 1, y), font, scale, bg, thickness)
    frame = cv2.putText(frame, text, (x + 1, y), font, scale, bg, thickness)
    frame = cv2.putText(frame, text, (x, y - 1), font, scale, bg, thickness)
    frame = cv2.putText(frame, text, (x, y + 1), font, scale, bg, thickness)
    frame = cv2.putText(frame, text, (x, y), font, scale, fg, thickness)
    return frame