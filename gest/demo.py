import argparse
import collections
import time

import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

from gest.cv_gui import text

img_width, img_height = 150, 150
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


parser = argparse.ArgumentParser()
parser.add_argument("model_path", help="Model path")
parser.add_argument("--camera", help="Camera index", type=int, default=0)
parser.add_argument("--history", help="History size", type=int, default=10)


HistoryEntry = collections.namedtuple('HistoryEntry', 'at score')


class App:

    def __init__(self, camera, model, history_size):
        self.camera = camera
        self.model = model
        self.history_size = history_size

    def run(self):
        video_capture = cv2.VideoCapture(self.camera)
        history = collections.deque()
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            now = time.time()
            score = model.predict(
                cv2.resize(frame, (img_width, img_height)).reshape((1, *input_shape)) / 255.
            )[0][0]
            history.append(HistoryEntry(now, score))
            display = np.array(frame[:, ::-1, :])
            if len(history) == self.history_size:
                fps = len(history) / (history[-1].at - history[0].at)
                avg_score = sum(e.score for e in history) / len(history)
                display = text(display, f'fps {fps:.0f}, score {avg_score:.0%}, avg over {len(history)}')
                history.popleft()
            cv2.imshow('Demo', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parser.parse_args()
    model = build_model()
    model.load_weights(args.model_path)
    App(
        camera=args.camera,
        model=model,
        history_size=args.history,
    ).run()
