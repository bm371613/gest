import collections
import pathlib
import time

import cv2
import numpy as np
import onnxruntime

from gest.threaded_pipeline import PipelineRun, Factory

DEFAULT_MODEL_FILE = pathlib.Path(__file__).parent / 'GES-120.onnx'
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406])[:, None, None]
IMAGE_STDDEV = np.asarray([0.229, 0.224, 0.225])[:, None, None]


class CvCameraInferencePipeline:

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
        self.camera = camera
        self.model_file = model_file

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

    @staticmethod
    def preprocessing(items):
        for item in items:
            frame = item.frame
            x = cv2.cvtColor(
                cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT)),
                cv2.COLOR_BGR2RGB,
            ).transpose((2, 0, 1)).astype(np.float32) / 255.
            x -= IMAGE_MEAN
            x /= IMAGE_STDDEV
            item.preprocessed = np.stack((x, np.flip(x, -1)))
            yield item

    def inference(self, items):
        session = onnxruntime.InferenceSession(str(self.model_file or DEFAULT_MODEL_FILE))
        for item in items:
            item.raw_inference_result = session.run(
                ['output'],
                {'input': item.preprocessed},
            )
            yield item

    @staticmethod
    def postprocessing(items):
        last_time = time.time()
        for item in items:
            heatmap, flipped_heatmap = item.raw_inference_result[0].squeeze(axis=1)
            item.inference_result = np.stack((heatmap, np.flip(flipped_heatmap, -1)))
            now = time.time()
            item.fps = 1 / (now - last_time)
            item.latency = now - item.captured_at
            yield item
            last_time = now

    def __call__(self):
        return PipelineRun(source=Factory(CvCameraInferencePipeline.Item), components=[
            self.video_capture,
            self.preprocessing,
            self.inference,
            self.postprocessing,
        ])()
