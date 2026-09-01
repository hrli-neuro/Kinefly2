# Kinefly2

ROS-free, Python 3, real-time tracker for tethered-fly kinematics: left/right
wing stroke amplitude (WSA), head yaw/roll, abdomen angle, and an auxiliary
intensity/wingbeat-frequency region, from one or two cameras at ~90 Hz, for
closed-loop experiments.

The image-processing core is ported from
[Kinefly](https://github.com/ssafarik/Kinefly) (Safarik, Dickinson Lab) via
[Benifly](https://github.com/bmslpsu/Benifly) (Cellini). The ROS layer is
replaced by a small threaded pipeline:

```
FLIR camera (bottom) ──> capture+track thread ──┐
                                                ├──> StateMerger ──> sinks:
FLIR camera (front)  ──> capture+track thread ──┘        │            • ZMQ JSON publisher (port 5556)
                                                         │            • HDF5 logger
     GUI (main thread): live view, draggable handles     │            • [future] Reiser LED-panel serial driver
```

Each update of the *primary* camera (the bottom view, which measures WSA)
produces one merged fly state, dispatched synchronously to all sinks — so a
future closed-loop output adds no queueing latency. Measured tracking cost is
well under 1 ms/frame; the total glass-to-state latency is dominated by camera
exposure + USB transfer (~1 frame at 90 Hz).

---

## 1. Installation from scratch

Works on Linux (rig) and macOS (development; cameras only via `opencv`/video).

```bash
# 1. Get the code
git clone https://github.com/hrli-neuro/Kinefly2.git
cd Kinefly2

# 2. Create the environment (Python 3.10 is pinned because FLIR's PySpin
#    wheels support a limited set of Python versions)
conda create -n kinefly2 python=3.10
conda activate kinefly2

# 3. Install dependencies
pip install numpy opencv-python pyyaml pyzmq h5py

# 4. Verify: run the test suite (generates a synthetic fly video and checks
#    tracked angles against ground truth; needs no camera)
python tests/test_tracking.py         # expects wings r=1.000, head r~0.99
python tests/test_dual_pipeline.py    # dual-camera + ZMQ + HDF5 architecture
```

### Additional install on the Linux rig (FLIR cameras)

1. Download from FLIR (flir.com → Spinnaker SDK):
   - the **Spinnaker SDK** .deb bundle for your Ubuntu version, and
   - the matching **spinnaker_python** (PySpin) wheel for **Python 3.10**.
2. Install the SDK debs (follow FLIR's README, including udev rules), then:
   ```bash
   conda activate kinefly2
   pip install spinnaker_python-*.whl
   python -c "import PySpin; print('PySpin OK')"
   ```
3. USB3 cameras need a larger kernel USB buffer (do once per boot, or make it
   permanent via the kernel cmdline as described in FLIR's install notes):
   ```bash
   sudo sh -c 'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb'
   ```

---

## 2. Running

### 2.1 Offline on a recorded video (setup, tuning, analysis; any OS)

```bash
python -m kinefly2.apps.run_offline config/camera_bottom.yaml --loop
```

Options: `--video FILE` (override the video in the config), `--pace` (play at
the video's frame rate), `--csv out.csv` (write per-frame angles),
`--no-gui`.

### 2.2 Real time on the rig (both cameras)

```bash
python -m kinefly2.apps.run_rig config/rig.yaml
```

- one live window per camera (disable with `--no-gui` for max headroom),
- merged fly state published as JSON on ZMQ `tcp://*:5556`,
- everything logged to `logs/flystate_YYYYMMDD_HHMMSS.h5` (disable `--no-log`),
- quit with `q` or the `exit` button.

Consume the live state from any language / any machine on the network:

```python
import zmq, json
sock = zmq.Context().socket(zmq.SUB)
sock.connect('tcp://localhost:5556')          # or tcp://<rig-ip>:5556
sock.setsockopt_string(zmq.SUBSCRIBE, '')
while True:
    s = json.loads(sock.recv_string())
    parts = s['cameras']['bottom']['parts']
    lmr = parts['left']['angles'][0] - parts['right']['angles'][0]  # L-R WSA (rad)
```

### 2.3 Timing validation

```bash
python -m kinefly2.apps.check_latency config/camera_bottom.yaml --seconds 30
```

Healthy rig numbers at 90 Hz: interval median ≈ 11.1 ms, dropped frames = 0,
track median in the low milliseconds.

### 2.4 GUI controls (same UX as Kinefly)

- **Handles (drag with mouse):** head/abdomen hinge points define the body
  axis; wing hinge points sit on the wing bases; for each part, the two line
  handles set the angular range (`angle_lo`/`angle_hi`) and the two circle
  handles set `radius_inner`/`radius_outer`; the aux ellipse has center and
  two radius handles.
- **Toolbar:** `exit`; `saveBG` (capture the current image as the background
  for subtraction — do it with the fly absent or not illuminated);
  `track: H A L R X` (toggle head/abdomen/left/right/aux);
  `subtract: H A LR X` (background subtraction per part); `invertcolor`
  (set if the fly is dark on a bright background); `stabilize`; `symmetric`
  (mirror left/right edits); `windows` (extra debug windows).
- **Every mouse-up saves the parameters back into that camera's YAML.**

---

## 3. Which parameters live in which files

| File | Contents |
|---|---|
| `config/rig.yaml` | Rig topology: camera list, primary camera, ZMQ port, log dir. |
| `config/camera_bottom.yaml` | Bottom view: `camera:` (capture settings) + `tracker:` (all tracking params, GUI-managed). |
| `config/camera_front.yaml` | Same, for the front view. |
| `config/background_<name>.png` | Per-camera background image, written by `saveBG`. |
| `kinefly2/gui/app.py` → `tracker_defaults()` | Defaults used for any tracker parameter missing from a camera YAML. |
| `logs/flystate_*.h5` | Per-experiment data (see layout in `kinefly2/output/logger.py`). |
| `tests/videos/*` | Auto-generated synthetic test fixtures — regenerated by the tests, don't tune configs here. |

### 3.1 `config/rig.yaml`

```yaml
cameras:                      # name -> per-camera YAML (paths relative to this file)
  bottom: camera_bottom.yaml
  front: camera_front.yaml
primary: bottom               # this camera's updates drive the merged output
use_gui: true
zmq_port: 5556                # remove/comment to disable the publisher
log_dir: ../logs              # remove/comment to disable HDF5 logging
```

### 3.2 `camera:` section of a camera YAML — capture settings

```yaml
camera:
  kind: pyspin        # pyspin (FLIR) | video (file) | opencv (USB webcam)
  # kind: pyspin
  serial: "23456789"  # camera serial number (SpinView shows it); omit = first camera
  fps: 90
  exposure_us: 2000   # short exposure: less blur, less latency
  gain_db: 10
  # kind: video
  path: ../tests/videos/synthetic_bottom.avi
  pace: false         # true = deliver frames at the video frame rate
  loop: false
  # kind: opencv
  index: 0            # OS camera index
```

### 3.3 `tracker:` section of a camera YAML — tracking parameters

Normally managed by the GUI; every key can also be edited by hand (restart to
apply). Anything missing falls back to `tracker_defaults()`.

**Top level**

| Key | Meaning |
|---|---|
| `scale_image` | Downscale factor applied to frames before tracking (0.5 halves resolution, ~4x faster). |
| `n_edges_max` | Max edges reported per wing by the edge tracker (1 = leading edge only). |
| `rc_background` | Time constant (s) of the moving-average background model. |
| `wingbeat_min/max` | Expected wingbeat frequency band (Hz) for the aux detector (it uses aliased spectra, so it works at 90 fps). |

**Per part** (`head:`, `abdomen:`, `left:`, `right:`, `aux:`)

| Key | Meaning |
|---|---|
| `tracker` | Algorithm: `edge` (radial edge angle — wings, or antennae for head yaw), `area` (rotation via polar image registration — head), `tip` (farthest point — abdomen), `intensity` (mean intensity + wingbeat frequency — aux). |
| `autozero` | Auto-center the reported angle on the resting position. |
| `threshold` | Edge-strength threshold (edge tracker); 0 = accept all. |
| `feathering` | Softening of mask edges (0–1). |
| `saturation_correction` | Compensate saturated pixels when background-subtracting. |

**Geometry, per part under `gui:`** — what the handles edit

| Key | Meaning |
|---|---|
| `track` | Whether this part is tracked (toolbar checkbox). |
| `subtract_bg` | Background subtraction for this part (needs a saved background). |
| `hinge: {x, y}` | Rotation center in image pixels (wing base, neck, abdomen base). |
| `radius_inner/outer` | Annulus radii (px) around the hinge that the tracker analyzes. |
| `angle_lo/hi` | Angular window (radians, in the body frame: 0 points from abdomen hinge toward head hinge). |
| `stabilize` | Image stabilization of the part (area tracker). |
| aux only: `center`, `radius1`, `radius2`, `angle` | Ellipse of the intensity region. |
| `gui.symmetric` | Mirror left/right wing edits in the GUI. |
| `gui.axis.pt1/pt2` | Optional body-axis tracker endpoints. |

### 3.4 Data file layout (`logs/*.h5`)

One row per merged state: `/merged/seq`, `/merged/t_host`, and per camera
`/<cam>/seq`, `t_camera`, `t_host`, `t_done`, `/<cam>/<part>_angle` and
`_radius` for head/abdomen/left/right, `/<cam>/aux_intensity`, `aux_freq`.
Angles are radians; times are seconds (host clock is monotonic; `t_camera` is
the device clock). Readable with h5py or MATLAB's `h5read`.

---

## 4. Adapting to a different rig

- **Different cameras:** change only the `camera:` sections. FLIR → set
  serials; other GenICam/USB vendors → use `kind: opencv`, or add a new
  `FrameSource` subclass in `kinefly2/camera/` (implement `open/read/close`
  returning `Frame` objects) and register it in `camera/base.py::make_source`.
- **One camera instead of two:** delete the second entry under `cameras:` in
  `rig.yaml` (or just run `run_offline` for a single camera).
- **Different views / body parts:** each camera YAML is independent — e.g. use
  the front camera's `left`/`right` edge trackers for wing-envelope roll edges
  and its `head` area tracker for head roll. Head yaw from below can use
  `tracker: edge` with a narrow annulus over the antennae instead of `area`.
- **Different frame rate:** set `fps` in the camera YAML; nothing else assumes
  90 Hz.
- **Different image scale/resolution:** tune hinges in the GUI (positions are
  stored unscaled in image pixels); use `scale_image` if tracking needs to be
  cheaper.
- **Fly appearance:** dark fly on bright background → `invertcolor` on.
  Uneven illumination → save a background and enable subtraction per part.
- **Network/ports:** `zmq_port` in `rig.yaml`; subscribers connect to the
  rig's IP.
- **New outputs (e.g. the LED-panel arena):** subclass `FlystateSink`
  (`kinefly2/output/base.py`), add it in `run_rig.py`. Sinks are called
  synchronously in the primary camera's thread — do sub-millisecond work
  inline, or enqueue to your own thread. The planned Reiser panels driver
  (serial protocol + Kinefly's coefficient mapping) is sketched in
  `config/rig.yaml`'s comments.

---

## 5. Repository layout

```
kinefly2/tracking/   Kinefly/Benifly image-processing core (trackers, polar transforms, GUI widgets)
kinefly2/camera/     FrameSource abstraction: pyspin_source, video_source, opencv_source
kinefly2/core/       CameraPipeline (per-camera thread), StateMerger, flystate dataclasses
kinefly2/output/     FlystateSink interface, ZMQ publisher, HDF5 logger; future panels driver
kinefly2/gui/        TrackerView: per-camera window, toolbar, draggable handles, YAML save
kinefly2/apps/       run_offline, run_rig, check_latency
config/              rig.yaml + per-camera YAMLs + saved backgrounds
tests/               synthetic-video generator + ground-truth and architecture tests
```

## 6. Status / roadmap

- [x] Python 3 port of the tracking core, verified against synthetic ground
      truth (wings r=1.000, head r=0.989).
- [x] Dual-camera threaded pipeline, merged state, ZMQ publisher, HDF5 logger.
- [ ] Live PySpin validation at 90 Hz on the rig (pending the Linux machine
      and FLIR cameras).
- [ ] Reiser LED-panel closed-loop sink (pending the arena). Verify the
      controller generation first: this targets the classic panels controller;
      G4 uses a different protocol.
