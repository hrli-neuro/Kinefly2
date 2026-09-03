"""FLIR/PointGrey camera source via the Spinnaker SDK (PySpin).

PySpin is imported lazily so the rest of the package runs on machines without
the SDK (e.g., macOS development). Install: download the Spinnaker SDK + the
spinnaker_python wheel matching the rig's Python version from flir.com.

Latency-critical settings:
- StreamBufferHandlingMode = NewestOnly: the default (OldestFirst) queues
  frames and silently adds latency whenever processing briefly falls behind.
- Manual exposure/gain: auto modes vary the exposure time frame-to-frame.

On Linux with two USB3 cameras, raise the USB-FS memory limit if enumeration
or streaming fails:  sudo sh -c 'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb'
"""
import time

import numpy as np

from .base import Frame, FrameSource


class PySpinSource(FrameSource):
    reconnectable = True

    def __init__(self, serial=None, fps=90.0, exposure_us=None, gain_db=None):
        self.serial = str(serial) if serial is not None else None
        self.fps = fps
        self.exposure_us = exposure_us
        self.gain_db = gain_db
        self.system = None
        self.cam = None
        self.frame_id = 0
        self.timeout_ms = 2000

    def open(self):
        import PySpin
        self.PySpin = PySpin
        self.system = PySpin.System.GetInstance()
        cam_list = self.system.GetCameras()
        try:
            if cam_list.GetSize() == 0:
                raise IOError('No FLIR cameras found')
            if self.serial:
                self.cam = cam_list.GetBySerial(self.serial)
            else:
                self.cam = cam_list.GetByIndex(0)
            self.cam.Init()
        finally:
            cam_list.Clear()

        nodemap = self.cam.GetNodeMap()
        stream = self.cam.GetTLStreamNodeMap()

        # Newest-only buffering: never process stale frames.
        handling = PySpin.CEnumerationPtr(stream.GetNode('StreamBufferHandlingMode'))
        handling.SetIntValue(handling.GetEntryByName('NewestOnly').GetValue())

        self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

        # Mono8 keeps transfer size minimal.
        try:
            self.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)
        except PySpin.SpinnakerException:
            pass

        if self.exposure_us is not None:
            self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
            self.cam.ExposureTime.SetValue(float(self.exposure_us))
        if self.gain_db is not None:
            self.cam.GainAuto.SetValue(PySpin.GainAuto_Off)
            self.cam.Gain.SetValue(float(self.gain_db))
        if self.fps is not None:
            try:
                self.cam.AcquisitionFrameRateEnable.SetValue(True)
                self.cam.AcquisitionFrameRate.SetValue(float(self.fps))
            except PySpin.SpinnakerException:
                pass  # Some models expose this differently; free-run instead.

        self.cam.BeginAcquisition()

    def read(self):
        PySpin = self.PySpin
        try:
            img = self.cam.GetNextImage(self.timeout_ms)
        except PySpin.SpinnakerException:
            return None
        try:
            if img.IsIncomplete():
                return self.read()
            arr = np.array(img.GetNDArray(), copy=True)
            t_camera = img.GetTimeStamp() * 1e-9    # Device clock, ns -> s.
            frame_id = img.GetFrameID()
        finally:
            img.Release()
        return Frame(image=arr, t_camera=t_camera, t_host=time.monotonic(),
                     frame_id=frame_id)

    def close(self):
        if self.cam is not None:
            try:
                self.cam.EndAcquisition()
            except Exception:
                pass
            self.cam.DeInit()
            del self.cam
            self.cam = None
        if self.system is not None:
            self.system.ReleaseInstance()
            self.system = None
