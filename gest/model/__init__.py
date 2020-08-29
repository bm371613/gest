import cv2
import numpy as np
import onnxruntime

IMAGE_WIDTH = 160
IMAGE_HEIGHT = 120


def load(model_file):
    return onnxruntime.InferenceSession(model_file)


def score(model, frame):
    x = cv2.cvtColor(
        cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT)),
        cv2.COLOR_BGR2RGB,
    ).transpose((2, 0, 1)).astype(np.float32) / 255.
    return model.run(['output'], {'input': x.reshape((1, *x.shape))})[0][0]
