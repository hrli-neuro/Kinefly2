"""Raw video recording per camera (MJPEG), on its own thread.

Frames are enqueued from the camera pipeline right after capture and written
by a background thread, so disk I/O never stalls tracking. If the queue fills
(disk too slow), frames are dropped and counted rather than blocking.

A sidecar CSV (<video>_frames.csv) records frame_id, t_camera, t_host for
every frame actually written, so video frame k can be matched exactly to the
HDF5 fly-state row with the same frame_id / seq.
"""
import queue
import threading

import cv2


class VideoRecorder:
    def __init__(self, path, fps, quality=95, sidecar_path=None, max_queue=600):
        self.path = str(path)
        self.fps = float(fps or 30.0)
        self.quality = int(quality)
        self.sidecar_path = sidecar_path
        self.queue = queue.Queue(maxsize=max_queue)
        self.writer = None
        self.sidecar = None
        self.n_written = 0
        self.n_dropped = 0
        self.closed = False
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name='video-recorder')
        self.thread.start()

    def write(self, frame):
        if self.closed:
            return
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            self.n_dropped += 1

    def _open(self, shape):
        h, w = shape[:2]
        self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*'MJPG'),
                                      self.fps, (w, h), isColor=False)
        if not self.writer.isOpened():
            raise IOError('Cannot open video writer: %s' % self.path)
        try:
            self.writer.set(cv2.VIDEOWRITER_PROP_QUALITY, self.quality)
        except Exception:
            pass
        if self.sidecar_path:
            self.sidecar = open(self.sidecar_path, 'w')
            self.sidecar.write('video_frame,frame_id,t_camera,t_host\n')

    def _run(self):
        while True:
            frame = self.queue.get()
            if frame is None:
                break
            if self.writer is None:
                self._open(frame.image.shape)
            self.writer.write(frame.image)
            if self.sidecar is not None:
                self.sidecar.write('%d,%d,%.6f,%.6f\n' % (
                    self.n_written, frame.frame_id, frame.t_camera, frame.t_host))
            self.n_written += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.queue.put(None)
        self.thread.join(timeout=30)
        if self.writer is not None:
            self.writer.release()
        if self.sidecar is not None:
            self.sidecar.close()
