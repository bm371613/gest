import pathlib
import time

import cv2
import numpy as np
import onnxruntime

from gest.pipeline import Pipeline, Factory

DEFAULT_MODEL_FILE = pathlib.Path(__file__).parent / 'GES-131.onnx'
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406])[:, None, None]
IMAGE_STDDEV = np.asarray([0.229, 0.224, 0.225])[:, None, None]


class InferenceSession:

    def __init__(self, model_file=None):
        self.onnx_inference_session = onnxruntime.InferenceSession(str(model_file or DEFAULT_MODEL_FILE))

    @staticmethod
    def cv2_preprocess(frame):
        x = cv2.cvtColor(
            cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT)),
            cv2.COLOR_BGR2RGB,
        ).transpose((2, 0, 1)).astype(np.float32) / 255.
        x -= IMAGE_MEAN
        x /= IMAGE_STDDEV
        return np.stack((x, np.flip(x, -1)))

    def onnx_run(self, input):
        return self.onnx_inference_session.run(['output'], {'input': input})

    @staticmethod
    def postprocess(output):
        heatmap, flipped_heatmap = output[0].squeeze(axis=1)
        return np.stack((heatmap, np.flip(flipped_heatmap, -1)))

    def cv2_run(self, frame):
        input = self.cv2_preprocess(frame)
        output = self.onnx_run(input)
        return self.postprocess(output)


class CvCameraInferencePipeline(Pipeline):

    class Item:
        def __init__(self):
            self.captured_at = None
            self.frame = None
            self.preprocessed = None
            self.raw_inference_result = None
            self.inference_result = None
            self.latency = None
            self.fps = None

    def __init__(self, camera=0, model_file=None):
        components = [
            self.video_capture,
            self.preprocessing,
            self.inference,
            self.postprocessing,
        ]
        super().__init__(components, default_input_factory=lambda: Factory(self.Item))
        self.camera = camera
        self.inference_session = InferenceSession(model_file)

    def video_capture(self, items):
        capture = cv2.VideoCapture(self.camera)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
        for item in items:
            item.captured_at = time.time()
            returned, item.frame = capture.read()
            if not returned:
                break
            yield item
        capture.release()

    def preprocessing(self, items):
        for item in items:
            item.preprocessed = self.inference_session.cv2_preprocess(item.frame)
            yield item

    def inference(self, items):
        for item in items:
            item.raw_inference_result = self.inference_session.onnx_run(item.preprocessed)
            yield item

    def postprocessing(self, items):
        last_time = time.time()
        for item in items:
            item.inference_result = self.inference_session.postprocess(item.raw_inference_result)
            now = time.time()
            item.fps = 1 / (now - last_time)
            item.latency = now - item.captured_at
            yield item
            last_time = now
