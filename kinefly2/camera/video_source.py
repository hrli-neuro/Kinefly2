"""Video-file frame source, for development without cameras and for offline analysis.

With pace=True, frames are delivered at the video's frame rate (or an fps
override) so the full real-time pipeline can be exercised; otherwise frames are
delivered as fast as they can be processed.
"""
import time

import cv2

from .base import Frame, FrameSource


class VideoFileSource(FrameSource):
    def __init__(self, path, fps=None, pace=False, loop=False):
        self.path = str(path)
        self.fps_override = fps
        self.pace = pace
        self.loop = loop
        self.cap = None
        self.fps = None
        self.frame_id = 0
        self.t_next = None

    def open(self):
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise IOError('Cannot open video file: %s' % self.path)
        self.fps = self.fps_override or self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def read(self):
        ret, img = self.cap.read()
        if not ret:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, img = self.cap.read()
            if not ret:
                return None

        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.pace:
            if self.t_next is None:
                self.t_next = time.monotonic()
            else:
                dt = self.t_next - time.monotonic()
                if dt > 0:
                    time.sleep(dt)
            self.t_next += 1.0 / self.fps

        t = time.monotonic()
        # The video's own timeline stands in for the camera clock.
        t_camera = self.frame_id / self.fps
        frame = Frame(image=img, t_camera=t_camera, t_host=t, frame_id=self.frame_id)
        self.frame_id += 1
        return frame

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
