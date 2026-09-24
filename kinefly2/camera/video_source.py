"""Video-file frame source, for development without cameras and for offline analysis.

With pace=True, frames are delivered at the video's frame rate (or an fps
override) so the full real-time pipeline can be exercised; otherwise frames are
delivered as fast as they can be processed.

seek() jumps to any frame -- the offline player's stepping and scrubbing -- and
median_background() builds a background from the video itself, for recordings
that have no fly-free frame.
"""
import time

import cv2
import numpy as np

from .base import Frame, FrameSource


class VideoFileSource(FrameSource):
    def __init__(self, path, fps=None, pace=False, loop=False):
        self.path = str(path)
        self.fps_override = fps
        self.pace = pace
        self.loop = loop
        self.cap = None
        self.fps = None
        self.n_frames = 0          # As the container reports it; 0 if it does not say.
        self.frame_id = 0
        self.t_next = None

    def open(self):
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise IOError('Cannot open video file: %s' % self.path)
        self.fps = self.fps_override or self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = max(0, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))

    def seek(self, index):
        """Make frame `index` the next one read() returns.

        The frame id and the video clock follow the position, so a frame
        reached by seeking is labelled exactly like the same frame reached by
        playing through to it."""
        index = max(0, int(index))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        self.frame_id = index
        self.t_next = None          # Pacing restarts from here.

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


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _sample_sequential(cap, n_samples):
    """Read the whole video, keeping ~n_samples frames spread over it without
    holding it all in memory: the sampling stride doubles whenever the kept set
    grows too big. For containers that cannot seek or do not know their length."""
    kept, stride, i = [], 1, 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % stride == 0:
            kept.append(_gray(img))
            if len(kept) >= 2 * n_samples:
                kept = kept[::2]
                stride *= 2
        i += 1
    return kept, i


def median_background(path, n_samples=41):
    """Temporal median of frames spread over the whole video.

    Anything that moves (wings, legs) vanishes and the static scene -- body,
    tether, illumination -- remains, which is the background to subtract when a
    recording has no frame without the fly. Frames are reached by seeking, so a
    long recording costs no more than a short one; a container that cannot seek
    or does not know its length is read through instead.

    Returns (image, n_frames_used, n_frames_in_video)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError('Cannot open video file: %s' % path)
    try:
        n = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        kept = []
        if n:
            want = np.unique(np.linspace(0, n - 1, min(n_samples, n)).round().astype(int))
            for idx in want:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, img = cap.read()
                if ok:
                    kept.append(_gray(img))
            if len(kept) < max(1, len(want) // 2):
                kept = []           # Seeking does not work here: read through.
        if not kept:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            kept, n = _sample_sequential(cap, n_samples)
    finally:
        cap.release()
    if not kept:
        raise IOError('No frames in the video: %s' % path)
    return np.median(np.stack(kept), axis=0).astype('uint8'), len(kept), n
