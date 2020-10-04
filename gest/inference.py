import cv2
import numpy as np
import onnxruntime

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406])[:, None, None]
IMAGE_STDDEV = np.asarray([0.229, 0.224, 0.225])[:, None, None]


class InferenceSession:

    def __init__(self, model_file):
        self.onnx_inference_session = onnxruntime.InferenceSession(model_file)

    def cv2_run(self, frame):
        x = cv2.cvtColor(
            cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT)),
            cv2.COLOR_BGR2RGB,
        ).transpose((2, 0, 1)).astype(np.float32) / 255.
        x = x.reshape((1, *x.shape))
        x -= IMAGE_MEAN
        x /= IMAGE_STDDEV
        return self.onnx_inference_session.run(['output'], {'input': x})[0][0][0]
