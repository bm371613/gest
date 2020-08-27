import argparse

import cv2
import pynput.keyboard

import gest.model
from gest.cv_gui import text


parser = argparse.ArgumentParser()
parser.add_argument("model_path", help="Model path")
parser.add_argument("--camera", help="Camera index", type=int, default=0)
parser.add_argument("--acc-weight", help="Camera index", type=float, default=1)

args = parser.parse_args()

video_capture = cv2.VideoCapture(args.camera)
model = gest.model.load(args.model_path)
acc_weight = args.acc_weight

keyboard = pynput.keyboard.Controller()
acc = 0
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    score = gest.model.score(model, frame)
    new_acc = (acc * acc_weight + score) / (acc_weight + 1)
    if acc <= .5 < new_acc:
        keyboard.press(pynput.keyboard.Key.shift)
    if acc >= .5 > new_acc:
        keyboard.release(pynput.keyboard.Key.shift)
    acc = new_acc
    cv2.imshow(
        'gest: shift',
        text(cv2.flip(frame, 1), f'score {score:.0%}, accumulated {acc:.0%}'),
    )
    if cv2.waitKey(1) == 27:  # esc to quit
        break
video_capture.release()
cv2.destroyAllWindows()
