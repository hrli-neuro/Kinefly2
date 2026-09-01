# Kinefly2

ROS-free, Python 3, real-time tracker for tethered-fly kinematics: wing stroke
amplitude, head yaw/roll, abdomen, and wingbeat-proxy intensity, from one or
two cameras at ~90 Hz, for closed-loop experiments.

The image-processing core is ported from
[Kinefly](https://github.com/ssafarik/Kinefly) (Safarik, Dickinson Lab) via
[Benifly](https://github.com/bmslpsu/Benifly) (Cellini), with the ROS layer
replaced by a small threaded pipeline: per-camera capture+track threads, a
state merger, and pluggable output sinks (ZMQ publisher, HDF5 logger, and —
when the arena is set up — a Reiser LED-panel serial driver).

## Setup

```bash
conda create -n kinefly2 python=3.10
conda activate kinefly2
pip install numpy opencv-python pyyaml pyzmq h5py
```

On the Linux rig, additionally install the FLIR Spinnaker SDK and the
`spinnaker_python` (PySpin) wheel matching the Python version, from flir.com.
For two USB3 cameras, raise the USB-FS limit:
`sudo sh -c 'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb'`

## Usage

**Offline / setup on a recorded video** (works on macOS, no camera needed):

```bash
python -m kinefly2.apps.run_offline config/camera_bottom.yaml --loop
```

Use the GUI exactly like Kinefly: drag the hinge points, inner/outer radius
and angle handles; toggle tracked parts (H/A/L/R/X) and background subtraction
in the toolbar; `saveBG` stores the background image; `q` or `exit` quits.
Edits are saved to the camera YAML on mouse-up, so a config tuned on video is
ready for the rig. `--csv out.csv` writes per-frame angles.

**Real-time, both cameras** (on the rig; also runs on video files for testing):

```bash
python -m kinefly2.apps.run_rig config/rig.yaml
```

Each update of the primary (bottom) camera produces a merged fly state that is
published as JSON on ZMQ (`tcp://*:5556` by default) and logged to HDF5 in
`log_dir`. Subscribe from any language:

```python
import zmq, json
sock = zmq.Context().socket(zmq.SUB)
sock.connect('tcp://localhost:5556')
sock.setsockopt_string(zmq.SUBSCRIBE, '')
state = json.loads(sock.recv_string())
wsa_lmr = (state['cameras']['bottom']['parts']['left']['angles'][0]
           - state['cameras']['bottom']['parts']['right']['angles'][0])
```

**Latency / throughput report:**

```bash
python -m kinefly2.apps.check_latency config/camera_bottom.yaml --seconds 10
```

## Configuration

- `config/rig.yaml` — camera list, primary camera, ZMQ port, log dir.
- `config/camera_bottom.yaml`, `config/camera_front.yaml` — per-camera:
  - `camera:` — capture settings (`kind: pyspin|video|opencv`, serial, fps,
    exposure, gain / video path).
  - `tracker:` — the full Kinefly parameter dict (hinges, radii, angle ranges,
    tracker types, thresholds). Managed by the GUI; editable by hand.

Tracker types per body part: `edge` (radial edge angles — wings),
`area` (rotation by polar image registration — head), `tip`, `intensity`.

## Tests

```bash
python tests/test_tracking.py       # tracking core vs synthetic ground truth
python tests/test_dual_pipeline.py  # dual camera + merger + ZMQ + HDF5
```

`tests/make_synthetic_video.py` generates the synthetic fly video, its
ground-truth CSV, and a matching tracker config.

## Roadmap

- [x] Python 3 port of the tracking core, verified against synthetic ground
      truth (wings r=1.000, head r=0.989).
- [x] Dual-camera threaded pipeline, merged state, ZMQ publisher, HDF5 logger.
- [ ] Live PySpin validation at 90 Hz on the rig (needs the new Linux machine
      and the FLIR cameras).
- [ ] Reiser LED-panel closed-loop sink (needs the arena): serial protocol and
      the Kinefly coefficient mapping are documented in the plan; it attaches
      via `kinefly2/output/base.py` without core changes.
