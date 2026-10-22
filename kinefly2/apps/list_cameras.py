"""List connected FLIR cameras and print the config block each one needs.

    python -m kinefly2.apps.list_cameras

The only thing Kinefly2 needs to identify a camera is its SERIAL NUMBER. That
is what binds a physical camera to `bottom` or `front` no matter which port it
is plugged into or what order the two enumerate in -- so never rely on index
order for a two-camera rig.

Run this on the rig with the cameras connected, then paste the printed block
into the `camera:` section of the matching camera YAML.

Requires the Spinnaker SDK and PySpin (see README section 1). Without them this
says so and exits cleanly, so it is harmless to run on a development machine.
"""
import argparse
import sys


def _node_value(nodemap, name, kind='int'):
    """Read a GenICam node, returning None when it is absent or unreadable.
    Cameras differ in which nodes they expose, so nothing here is assumed."""
    import PySpin

    try:
        if kind == 'int':
            node = PySpin.CIntegerPtr(nodemap.GetNode(name))
        elif kind == 'float':
            node = PySpin.CFloatPtr(nodemap.GetNode(name))
        elif kind == 'enum':
            node = PySpin.CEnumerationPtr(nodemap.GetNode(name))
        else:
            node = PySpin.CStringPtr(nodemap.GetNode(name))
        if not PySpin.IsAvailable(node) or not PySpin.IsReadable(node):
            return None
        if kind == 'enum':
            return node.GetCurrentEntry().GetSymbolic()
        return node.GetValue()
    except Exception:
        return None


def _node_max(nodemap, name, kind='int'):
    import PySpin

    try:
        if kind == 'int':
            node = PySpin.CIntegerPtr(nodemap.GetNode(name))
        else:
            node = PySpin.CFloatPtr(nodemap.GetNode(name))
        if not PySpin.IsAvailable(node):
            return None
        return node.GetMax()
    except Exception:
        return None


def describe(cam):
    """Everything worth knowing about one camera, without claiming it for long."""
    import PySpin

    tl = cam.GetTLDeviceNodeMap()
    info = {
        'serial': _node_value(tl, 'DeviceSerialNumber', 'str'),
        'model': _node_value(tl, 'DeviceModelName', 'str'),
        'vendor': _node_value(tl, 'DeviceVendorName', 'str'),
        'interface': _node_value(tl, 'DeviceCurrentSpeed', 'enum') or
                     _node_value(tl, 'DeviceType', 'enum'),
    }
    try:
        cam.Init()
    except PySpin.SpinnakerException as e:
        # Almost always means another process (SpinView, or a running
        # Kinefly2) already holds it.
        info['error'] = 'in use or unavailable (%s)' % str(e).split('\n')[0]
        return info
    try:
        nm = cam.GetNodeMap()
        info['width'] = _node_value(nm, 'Width')
        info['height'] = _node_value(nm, 'Height')
        # WidthMax's value is the sensor width; its own maximum is just the
        # node's type limit (65535).
        info['width_max'] = _node_value(nm, 'WidthMax') or _node_max(nm, 'Width')
        info['height_max'] = _node_value(nm, 'HeightMax') or _node_max(nm, 'Height')
        info['fps'] = _node_value(nm, 'AcquisitionFrameRate', 'float')
        # The ceiling at the CURRENT ROI: shrink the ROI and this goes up,
        # which is usually how a USB3 pair reaches 90 Hz. First-generation
        # cameras have no resulting-rate node; the rate node's maximum is the
        # same ceiling.
        info['fps_max'] = (_node_value(nm, 'AcquisitionResultingFrameRate', 'float') or
                           _node_max(nm, 'AcquisitionFrameRate', 'float'))
        info['exposure_us'] = _node_value(nm, 'ExposureTime', 'float')
        info['gain_db'] = _node_value(nm, 'Gain', 'float')
    finally:
        cam.DeInit()
    return info


def config_block(info, fps=90.0):
    """A ready-to-paste `camera:` section for this camera."""
    exposure = int(min(info.get('exposure_us') or 2000, 0.8e6 / fps))
    lines = [
        'camera:',
        '  kind: pyspin',
        '  serial: "%s"          # %s' % (info.get('serial'), info.get('model') or ''),
        '  fps: %g' % fps,
        '  exposure_us: %d          # must be < %d us at %g fps' % (
            exposure, int(1e6 / fps), fps),
        '  gain_db: %g' % (info.get('gain_db') or 0),
    ]
    fps_max = info.get('fps_max')
    if fps_max and fps_max < 0.99 * fps:
        lines.insert(4, '  # reaches only %.1f fps at its current crop: set a smaller roi to go'
                        ' faster,\n  # or Kinefly2 runs at whatever maximum the camera reports'
                        % fps_max)
    w, h = info.get('width_max'), info.get('height_max')
    if w and h:
        lines += [
            '  # Crop to reach the frame rate; centred when the offsets are omitted.',
            '  # roi: {width: %d, height: %d}' % (w // 2, h // 2),
        ]
    lines += [
        '  # Hardware sync for a two-camera rig: one primary strobes, the other',
        '  # follows, so both views share a frame clock.',
        '  # trigger: {mode: primary}      # on the other camera: {mode: secondary}',
    ]
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fps', type=float, default=90.0,
                    help='Frame rate to put in the printed config block (default 90)')
    args = ap.parse_args()

    try:
        import PySpin
    except ImportError:
        print('PySpin (Spinnaker SDK) is not installed here, so no cameras can be\n'
              'enumerated. That is expected on a development machine -- see README\n'
              'section 1, "Additional install on the Linux rig".')
        return 0

    system = PySpin.System.GetInstance()
    v = system.GetLibraryVersion()
    print('Spinnaker %d.%d.%d.%d' % (v.major, v.minor, v.type, v.build))
    cam_list = system.GetCameras()
    try:
        n = cam_list.GetSize()
        if n == 0:
            print('\nNo FLIR cameras found. Check the cable and, on Linux, that the\n'
                  'udev rules from the SDK install are in place. USB3 cameras also\n'
                  'need the larger kernel buffer:\n'
                  "  sudo sh -c 'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb'")
            return 0
        print('%d camera%s found\n' % (n, '' if n == 1 else 's'))
        for i in range(n):
            info = describe(cam_list.GetByIndex(i))
            print('=' * 62)
            print('[%d] %s %s   serial %s' % (i, info.get('vendor') or '',
                                              info.get('model') or '?',
                                              info.get('serial') or '?'))
            if info.get('error'):
                print('    %s' % info['error'])
                print('    Close SpinView or any running Kinefly2 and try again.')
                continue
            if info.get('interface'):
                print('    link:       %s' % info['interface'])
                if info['interface'] in ('HighSpeed', 'FullSpeed'):
                    print('                USB 2 speed: use a USB 3 port and cable')
            if info.get('width'):
                print('    resolution: %sx%s  (sensor max %sx%s)'
                      % (info['width'], info['height'],
                         info.get('width_max'), info.get('height_max')))
            if info.get('fps_max'):
                print('    max fps:    %.1f at the current ROI' % info['fps_max'])
            print()
            print(config_block(info, fps=args.fps))
            print()
        print('=' * 62)
        print('Paste the block for each camera into the `camera:` section of its\n'
              'YAML (config/camera_bottom.yaml, config/camera_front.yaml), replacing\n'
              'the `kind: video` block. Nothing else changes: the tracking, GUI,\n'
              'logging and ZMQ paths are the same whether frames come from a file\n'
              'or a camera.')
    finally:
        cam_list.Clear()
        system.ReleaseInstance()
    return 0


if __name__ == '__main__':
    sys.exit(main())
