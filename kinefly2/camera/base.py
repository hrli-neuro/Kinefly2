"""Camera abstraction: every image source yields Frame objects with timestamps.

Sources: 'pyspin' (FLIR/Spinnaker, Linux rig), 'video' (file playback, for
development and offline analysis), 'opencv' (generic USB camera).
"""
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    image: np.ndarray      # Grayscale uint8, shape (h, w).
    t_camera: float        # Device timestamp in seconds (source clock; NaN if unavailable).
    t_host: float          # time.monotonic() at receipt on the host.
    frame_id: int          # Monotonically increasing; gaps indicate dropped frames.


class FrameSource(ABC):
    """Blocking frame source. read() returns the next Frame, or None at end of stream."""

    reconnectable = False   # Live cameras: None from read() means "reconnect", not EOF.

    @abstractmethod
    def open(self):
        ...

    @abstractmethod
    def read(self):
        ...

    @abstractmethod
    def close(self):
        ...

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()


def make_source(config):
    """Build a FrameSource from a camera config dict (the 'camera' section of a
    per-camera YAML). config['kind'] selects the implementation."""
    kind = config.get('kind', 'video')
    if kind == 'video':
        from .video_source import VideoFileSource
        return VideoFileSource(config['path'],
                               fps=config.get('fps'),
                               pace=config.get('pace', False),
                               loop=config.get('loop', False))
    elif kind == 'opencv':
        from .opencv_source import OpenCVSource
        return OpenCVSource(config.get('index', 0), fps=config.get('fps'))
    elif kind == 'pyspin':
        from .pyspin_source import PySpinSource
        return PySpinSource(serial=config.get('serial'),
                            fps=config.get('fps', 90.0),
                            exposure_us=config.get('exposure_us'),
                            gain_db=config.get('gain_db'))
    else:
        raise ValueError('Unknown camera kind: %r' % kind)


def now():
    return time.monotonic()
