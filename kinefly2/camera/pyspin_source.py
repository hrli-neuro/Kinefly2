"""FLIR/PointGrey camera source via the Spinnaker SDK (PySpin).

PySpin is imported lazily so the rest of the package runs on machines without
the SDK (e.g., macOS development). Install: download the Spinnaker SDK + the
spinnaker_python wheel matching the rig's Python version from flir.com.

Latency-critical settings:
- StreamBufferHandlingMode = NewestOnly: the default (OldestFirst) queues
  frames and silently adds latency whenever processing briefly falls behind.
- Manual exposure/gain: auto modes vary the exposure time frame-to-frame.

FRAME RATE. `fps` is a request. A camera that cannot reach it (crop too large
for the USB link, exposure longer than the frame interval) is set to its
maximum instead, and `self.fps` is then what the camera reports it will
actually deliver -- which is what the video recorder writes into the AVI
header and what the staleness threshold is derived from. Trusting the request
would label a 72 Hz video as 90 Hz and stretch every re-analysis time axis.

On Linux with two USB3 cameras, raise the USB-FS memory limit if enumeration
or streaming fails:  sudo sh -c 'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb'
"""
import time

import numpy as np

from .base import Frame, FrameSource


class PySpinSource(FrameSource):
    reconnectable = True

    def __init__(self, serial=None, fps=90.0, exposure_us=None, gain_db=None,
                 roi=None, binning=None, pixel_format='Mono8', trigger=None):
        self.serial = str(serial) if serial is not None else None
        self.fps_requested = fps     # What the config asked for.
        self.fps = fps               # What the camera delivers, once open() has read it back.
        self.exposure_us = exposure_us
        self.gain_db = gain_db
        self.roi = dict(roi) if roi else None          # {width, height, offset_x, offset_y}
        self.binning = int(binning) if binning else None
        self.pixel_format = pixel_format or 'Mono8'
        self.trigger = dict(trigger) if trigger else None   # {mode, line, source}
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
            fmt = getattr(PySpin, 'PixelFormat_' + self.pixel_format)
            self.cam.PixelFormat.SetValue(fmt)
        except (PySpin.SpinnakerException, AttributeError):
            pass

        # Binning before ROI: binning changes what the width/height limits mean.
        if self.binning:
            for node in ('BinningHorizontal', 'BinningVertical'):
                try:
                    getattr(self.cam, node).SetValue(self.binning)
                except PySpin.SpinnakerException:
                    pass
        self._apply_roi()
        self._apply_trigger()

        if self.exposure_us is not None:
            self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
            self.cam.ExposureTime.SetValue(float(self.exposure_us))
        if self.gain_db is not None:
            self.cam.GainAuto.SetValue(PySpin.GainAuto_Off)
            self.cam.Gain.SetValue(float(self.gain_db))
        # Last: ROI, binning and exposure all lower the rate the camera can reach.
        self._apply_frame_rate()

        self.cam.BeginAcquisition()

    def _apply_frame_rate(self):
        """Request `fps_requested`, capped at what the camera can do, then read
        back the rate it will actually deliver into `self.fps`.

        A secondary camera's frames come from the primary's trigger, so its
        nominal rate stays the requested one (it must match the primary); the
        read-back only tells whether it can keep up."""
        PySpin = self.PySpin
        want = self.fps_requested
        if want is not None:
            try:
                self.cam.AcquisitionFrameRateEnable.SetValue(True)
                node = self.cam.AcquisitionFrameRate
                node.SetValue(min(float(want), float(node.GetMax())))
            except (PySpin.SpinnakerException, AttributeError) as e:
                print('camera %s: could not set the frame rate (%s); it free-runs'
                      % (self.serial or '', str(e).split('\n')[0]))
        actual = self._read_frame_rate()
        secondary = (self.trigger or {}).get('mode') == 'secondary'
        if want is not None and (secondary or actual is None):
            self.fps = float(want)
        else:
            self.fps = actual
        if want is not None and actual is not None and actual < 0.99 * float(want):
            frame_us = 1e6 / float(want)
            if self.exposure_us and float(self.exposure_us) >= frame_us:
                why = 'exposure_us %g is longer than a %g fps frame (%.0f us)' % (
                    self.exposure_us, want, frame_us)
            else:
                why = 'usually the crop (roi) is too large for the link; shrink it'
            if secondary:
                print('camera %s: can reach only %.1f fps, below the %g fps trigger it '
                      'must follow (%s)' % (self.serial or '', actual, want, why))
            else:
                print('camera %s: reaches only %.1f fps, not the %g requested -- running '
                      'at %.1f (%s)' % (self.serial or '', actual, want, actual, why))
        return self.fps

    def _read_frame_rate(self):
        """The frame rate the camera says it will deliver with the current
        settings, or None when it does not say."""
        PySpin = self.PySpin
        for name in ('AcquisitionResultingFrameRate', 'AcquisitionFrameRate'):
            try:
                value = float(getattr(self.cam, name).GetValue())
            except (PySpin.SpinnakerException, AttributeError):
                continue
            if value > 0:
                return value
        return None

    def _apply_roi(self):
        """Crop the sensor. This is usually what buys the frame rate: a smaller
        ROI means less to move over the bus, so two USB3 cameras can both hold
        90 Hz where full frames cannot. Offsets default to centred.

        Width/height must land on the sensor's increment, so each value is
        snapped down rather than rejected."""
        PySpin = self.PySpin
        if not self.roi:
            return

        def snap(node, want):
            inc, lo, hi = node.GetInc(), node.GetMin(), node.GetMax()
            want = max(lo, min(int(want), hi))
            return lo + ((want - lo) // inc) * inc

        try:
            # Offsets to 0 first, or a large existing offset caps the new width.
            for name in ('OffsetX', 'OffsetY'):
                try:
                    getattr(self.cam, name).SetValue(0)
                except PySpin.SpinnakerException:
                    pass
            if self.roi.get('width'):
                self.cam.Width.SetValue(snap(self.cam.Width, self.roi['width']))
            if self.roi.get('height'):
                self.cam.Height.SetValue(snap(self.cam.Height, self.roi['height']))
            ox = self.roi.get('offset_x')
            oy = self.roi.get('offset_y')
            if ox is None:
                ox = (self.cam.Width.GetMax() - self.cam.Width.GetValue()) // 2
            if oy is None:
                oy = (self.cam.Height.GetMax() - self.cam.Height.GetValue()) // 2
            self.cam.OffsetX.SetValue(snap(self.cam.OffsetX, ox))
            self.cam.OffsetY.SetValue(snap(self.cam.OffsetY, oy))
        except PySpin.SpinnakerException as e:
            raise IOError('ROI %r rejected by the camera: %s' % (self.roi, e))

    def _apply_trigger(self):
        """Hardware sync for a two-camera rig.

        Free-running cameras drift against each other, so a frame from one can
        be up to a frame away from the other's. That matters here because the
        two views are compared directly -- one camera's displacement channel is
        calibrated against the other's angle -- so they should share a clock.

        mode 'primary'   free-runs and strobes its exposure on an output line;
        mode 'secondary' takes each frame from that line;
        mode 'off'       (or no trigger block) is plain free-running.
        """
        PySpin = self.PySpin
        mode = (self.trigger or {}).get('mode', 'off')
        if mode in (None, 'off', False):
            try:
                self.cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
            except PySpin.SpinnakerException:
                pass
            return
        if mode == 'primary':
            # Expose-active strobe on the chosen output line.
            line = int((self.trigger or {}).get('line', 2))
            try:
                self.cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
                self.cam.LineSelector.SetValue(getattr(PySpin, 'Line%d' % line))
                self.cam.LineMode.SetValue(PySpin.LineMode_Output)
            except (PySpin.SpinnakerException, AttributeError) as e:
                raise IOError('cannot configure primary strobe on Line%d: %s' % (line, e))
            return
        if mode == 'secondary':
            source = (self.trigger or {}).get('source', 'Line3')
            try:
                self.cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
                self.cam.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)
                self.cam.TriggerSource.SetValue(getattr(PySpin, 'TriggerSource_' + source))
                self.cam.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)
                self.cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
            except (PySpin.SpinnakerException, AttributeError) as e:
                raise IOError('cannot configure secondary trigger on %s: %s' % (source, e))
            return
        raise ValueError("trigger mode must be 'off', 'primary' or 'secondary', got %r" % mode)

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
