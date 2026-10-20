"""The frame rate Kinefly2 uses is the one the camera delivers, not the one
the config asked for.

A camera that cannot reach `fps` (crop too large for the link, exposure longer
than a frame) used to swallow the error and free-run at an unknown rate while
everything downstream -- the AVI header, the staleness threshold -- still
believed the request. These tests drive PySpinSource against a fake camera,
so they need neither PySpin nor hardware.
"""
import contextlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from kinefly2.apps import list_cameras
from kinefly2.camera.pyspin_source import PySpinSource


class SpinnakerException(Exception):
    pass


class FakePySpin:
    SpinnakerException = SpinnakerException


class Node:
    def __init__(self, value=None, maximum=None, readable=True):
        self.value, self.maximum, self.readable = value, maximum, readable

    def GetValue(self):
        if not self.readable:
            raise SpinnakerException('node not readable')
        return self.value

    def SetValue(self, value):
        if self.maximum is not None and value > self.maximum:
            raise SpinnakerException('value out of range')
        self.value = value

    def GetMax(self):
        return self.maximum


class FakeCam:
    """A camera whose crop and exposure allow at most `max_fps`."""

    def __init__(self, max_fps, resulting_readable=True, rate_readable=True):
        self.max_fps = max_fps
        self.resulting_readable = resulting_readable
        self.AcquisitionFrameRateEnable = Node(False)
        self.AcquisitionFrameRate = Node(30.0, max_fps, readable=rate_readable)

    @property
    def AcquisitionResultingFrameRate(self):
        return Node(min(self.AcquisitionFrameRate.value, self.max_fps),
                    readable=self.resulting_readable)


def open_fake(max_fps, fps=90, cam_kw=None, **src_kw):
    """Run the frame-rate step against a fake camera; returns (source, printed)."""
    src = PySpinSource(serial='123', fps=fps, **src_kw)
    src.PySpin, src.cam = FakePySpin, FakeCam(max_fps, **(cam_kw or {}))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        src._apply_frame_rate()
    return src, out.getvalue()


def test_rate_reached():
    src, printed = open_fake(max_fps=120)
    assert src.fps == 90 and src.fps_requested == 90, src.fps
    assert src.cam.AcquisitionFrameRate.value == 90
    assert printed == '', printed
    print('camera reaches 90: runs at 90, nothing printed')


def test_rate_capped_and_read_back():
    """The case the old code got wrong: the request is out of range."""
    src, printed = open_fake(max_fps=72.3)
    assert src.cam.AcquisitionFrameRate.value == 72.3, 'request must be capped, not dropped'
    assert src.fps == 72.3, 'downstream must see the delivered rate, got %r' % src.fps
    assert 'reaches only 72.3 fps' in printed and 'roi' in printed, printed
    print('camera tops out at 72.3: set to 72.3, fps reads back 72.3, warned')


def test_exposure_named_as_the_cause():
    src, printed = open_fake(max_fps=48, exposure_us=20000)
    assert src.fps == 48
    assert 'exposure_us 20000' in printed, printed
    print('exposure longer than a frame is named as the reason')


def test_fallbacks_when_the_camera_does_not_say():
    # No resulting-rate node: the frame-rate node itself (already capped).
    src, _ = open_fake(max_fps=72.3, cam_kw={'resulting_readable': False})
    assert src.fps == 72.3, src.fps
    # Nothing readable at all: the request is all there is.
    src, _ = open_fake(max_fps=72.3, cam_kw={'resulting_readable': False,
                                             'rate_readable': False})
    assert src.fps == 90, src.fps
    print('read-back falls back to AcquisitionFrameRate, then to the request')


def test_secondary_keeps_the_trigger_rate():
    """A triggered camera runs at the primary's rate; the read-back only says
    whether it can keep up."""
    src, printed = open_fake(max_fps=120, trigger={'mode': 'secondary'})
    assert src.fps == 90 and printed == '', (src.fps, printed)
    src, printed = open_fake(max_fps=60, trigger={'mode': 'secondary'})
    assert src.fps == 90, src.fps
    assert 'below the 90 fps trigger' in printed, printed
    print('secondary: nominal rate stays the trigger rate; warned when it cannot follow')


def test_list_cameras_flags_a_slow_camera():
    info = {'serial': '1', 'model': 'X', 'gain_db': 0, 'fps_max': 61.0}
    block = list_cameras.config_block(info, fps=90)
    assert 'reaches only 61.0 fps' in block, block
    assert 'reaches only' not in list_cameras.config_block(dict(info, fps_max=160.0), fps=90)
    print('list_cameras notes a camera that cannot reach the requested rate')


if __name__ == '__main__':
    test_rate_reached()
    test_rate_capped_and_read_back()
    test_exposure_named_as_the_cause()
    test_fallbacks_when_the_camera_does_not_say()
    test_secondary_keeps_the_trigger_rate()
    test_list_cameras_flags_a_slow_camera()
    print('OK')
