"""Generic USB/UVC camera via OpenCV VideoCapture. No device timestamps."""
import time

import cv2
import numpy as np

from .base import Frame, FrameSource


class OpenCVSource(FrameSource):
    reconnectable = True

    def __init__(self, index=0, fps=None):
        self.index = index
        self.fps = fps
        self.cap = None
        self.frame_id = 0

    def open(self):
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            raise IOError('Cannot open camera index %r' % self.index)
        if self.fps:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Keep the driver-side buffer minimal so frames are fresh.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        ret, img = self.cap.read()
        if not ret:
            return None
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frame = Frame(image=img, t_camera=np.nan, t_host=time.monotonic(),
                      frame_id=self.frame_id)
        self.frame_id += 1
        return frame

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
