"""Latency/throughput report: runs one camera headless and prints timing
percentiles for the stages the software controls.

    python -m kinefly2.apps.check_latency config/camera_bottom.yaml --seconds 10

Stages reported per frame:
  interval   time between successive frame arrivals (should be ~1/fps)
  track      fly.update duration
  total      frame arrival -> state ready (what a closed-loop sink would add to)
Camera-side latency (exposure + transfer) shows up as t_host - t_camera on the
rig once PySpin provides device timestamps; it needs a clock-offset calibration
and is reported separately there.
"""
import argparse
import sys
import time

import numpy as np
import yaml

from ..camera.base import make_source
from ..gui.app import TrackerView


def pct(a):
    a = np.asarray(a) * 1e3
    return 'median %6.2f ms   p90 %6.2f ms   p99 %6.2f ms   max %6.2f ms' % (
        np.percentile(a, 50), np.percentile(a, 90), np.percentile(a, 99), a.max())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('config')
    ap.add_argument('--seconds', type=float, default=10.0)
    ap.add_argument('--name', default='latency-test')
    args = ap.parse_args()

    with open(args.config) as f:
        cam_cfg = (yaml.safe_load(f) or {}).get('camera') or {}

    view = TrackerView(args.name, args.config, use_gui=False)
    source = make_source(cam_cfg)
    source.open()

    intervals, tracks, totals, frame_ids = [], [], [], []
    t_end = time.monotonic() + args.seconds
    t_prev = None
    try:
        while time.monotonic() < t_end:
            frame = source.read()
            if frame is None:
                break
            state = view.process(frame)
            t_done = time.monotonic()
            if t_prev is not None:
                intervals.append(frame.t_host - t_prev)
            t_prev = frame.t_host
            frame_ids.append(frame.frame_id)
            if state is not None:
                tracks.append(view.dtTrack)
                totals.append(t_done - frame.t_host)
    finally:
        source.close()

    if not tracks:
        print('No frames processed.')
        return 1

    print('frames processed: %d' % len(tracks))
    ids = np.asarray(frame_ids)
    dropped = int((np.diff(ids) - 1).sum()) if len(ids) > 1 else 0
    print('dropped frames (frame_id gaps): %d' % dropped)
    if intervals:
        print('interval: ' + pct(intervals))
    print('track:    ' + pct(tracks))
    print('total:    ' + pct(totals))
    return 0


if __name__ == '__main__':
    sys.exit(main())
