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

**Backgrounds.** For faint wing blur on a bright backlight, background
subtraction is essential. Two ways to get a background:
- On the rig: click `saveBG` before the fly is mounted (or with the fly not
  illuminated).
- From an existing video: `--make-bg` computes the temporal median of the
  whole video (moving wings/legs vanish, the static body/tether stay) and
  saves it as this camera's background, then exits:
  ```bash
  python -m kinefly2.apps.run_offline config/camera_bottom.yaml --make-bg
  ```
  Then enable the `subtract` checkboxes (or set `subtract_bg: true` per part).

**Independent left/right wing regions.** The `symmetric` toolbar checkbox
controls whether edits mirror across the body axis. Checked, moving one
wing's hinge/radius/angle handles moves the other's to match (fast initial
setup). Unchecked, every left/right handle is fully independent — use this
when the two wings need different scanning regions. The setting persists in
the YAML (`gui.symmetric`).

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

### 2.3 Per-experiment recordings (open- or closed-loop)

Everything that changes between recordings lives in one experiment YAML
(`experiments/default.yaml`: data root, genotype, fly id, protocol, notes,
video on/off, event port) with command-line overrides for quick successive
flies:

```bash
python -m kinefly2.apps.run_rig config/rig.yaml --exp experiments/default.yaml --fly 018
python -m kinefly2.apps.run_rig config/rig.yaml --exp experiments/default.yaml --fly 019 --protocol OL_stripe --no-video
python -m kinefly2.apps.run_rig config/rig.yaml --exp experiments/default.yaml --cameras bottom   # front camera off
```

Each recording gets a folder named with the lab convention
`{date}_{genotype}_{fly_id}_{protocol}` (suffix `_2`, `_3`… if it exists):

```
<data_root>/<date>/260819_XHCS_XHCS_017_YAMV/
  260819_XHCS_XHCS_017_YAMV_flystate.h5       angles + timestamps (+ /events)
  260819_XHCS_XHCS_017_YAMV_bottom.avi        raw MJPEG video (record_video: true)
  260819_XHCS_XHCS_017_YAMV_bottom_frames.csv frame_id/t_camera/t_host per video frame
  260819_XHCS_XHCS_017_YAMV_front.avi / _front_frames.csv
  260819_XHCS_XHCS_017_YAMV_events.jsonl      stimulus events as received
  260819_XHCS_XHCS_017_YAMV_experiment.yaml   resolved settings after overrides
  configs/                                    snapshot of rig.yaml + camera YAMLs used
```

The raw video makes every recording re-analyzable with different regions or
trackers: `run_offline <folder>/configs/camera_bottom.yaml --video <folder>/<name>_bottom.avi --csv out.csv`.
Video frame *k* corresponds to the sidecar row *k*, whose `frame_id` equals
`/<camera>/seq` in the HDF5.

**Stimulus events on the same clock.** Your arena/stimulus program can push
any string (JSON recommended) to the recording; it is stamped on arrival with
the fly-state clock (`t_host`) and wall-clock time (`t_wall`) and written to
`/events` in the HDF5 and to the `.jsonl`:

```python
import zmq
sock = zmq.Context().socket(zmq.PUSH)
sock.connect('tcp://localhost:5557')             # events_port; rig IP from another machine
sock.send_string('{"trial": 12, "event": "closed_loop_start", "gain": 1.5}')
```

Without `--exp` (or any experiment flag) `run_rig` behaves as before: an
auto-named `flystate_<timestamp>.h5` in `rig.yaml`'s `log_dir`, no video.

### 2.4 Live browser dashboard

```bash
python -m kinefly2.apps.dashboard                       # tracker on this machine
python -m kinefly2.apps.dashboard --zmq tcp://<rig-ip>:5556 --port 8000
```

Open `http://<machine>:8000` from any computer on the network for live
traces of L, R, L−R, head yaw/roll and wingbeat frequency. It is a separate
process reading the ZMQ stream, so it adds no load to the tracker, and the
page has no external dependencies (works on an offline rig network).

### 2.5 Timing validation

```bash
python -m kinefly2.apps.check_latency config/camera_bottom.yaml --seconds 30
```

Healthy rig numbers at 90 Hz: interval median ≈ 11.1 ms, dropped frames = 0,
track median in the low milliseconds.

### 2.6 GUI controls (same UX as Kinefly)

- **Handles (drag with mouse):** head/abdomen hinge points define the body
  axis; wing hinge points sit on the wing bases; for each part, the two line
  handles set the angular range (`angle_lo`/`angle_hi`) and the two circle
  handles set `radius_inner`/`radius_outer`; the aux ellipse has center and
  two radius handles.
- **Toolbar:** `exit`; `saveBG` (capture the current image as the background
  for subtraction — do it with the fly absent or not illuminated);
  `autoplace` (place all regions automatically from the current image, see
  below);
  `track: H A L R X` (toggle head/abdomen/left/right/aux);
  `subtract: H A LR X` (background subtraction per part); `invertcolor`
  (set if the fly is dark on a bright background); `stabilize`; `symmetric`
  (mirror left/right edits); `windows` (extra debug windows).
- **Every mouse-up saves the parameters back into that camera's YAML** and
  stores the current frame as `config/reference_<camera>.png`, the frame the
  regions were tuned on.

**Auto-placement (`autoplace`).** For a new fly, click `autoplace`: the
current frame is registered to the saved reference frame (rotation +
translation, ECC) and every hinge/center moves accordingly, so a fly mounted
a few pixels or degrees off needs at most a nudge. The status line reports
the shift/rotation found. If no reference exists yet, a rough silhouette
bootstrap (bottom view) places the body axis and hinges from the fly's
outline — refine once, save, and the registration method takes over from
then on. Each camera has its own reference.

---

## 3. Which parameters live in which files

| File | Contents |
|---|---|
| `config/rig.yaml` | Rig topology: camera list, primary camera, ZMQ port, log dir. |
| `config/camera_bottom.yaml` | Bottom view: `camera:` (capture settings) + `tracker:` (all tracking params, GUI-managed). |
| `config/camera_front.yaml` | Same, for the front view. |
| `config/background_<name>.png` | Per-camera background image, written by `saveBG` or `--make-bg`. |
| `config/reference_<name>.png` | Per-camera reference frame for `autoplace` (saved with the params). |
| `experiments/*.yaml` | Per-recording settings: data root, genotype/fly/protocol, notes, video on/off, event port (section 2.3). |
| `kinefly2/gui/app.py` → `tracker_defaults()` | Defaults used for any tracker parameter missing from a camera YAML. |
| `<data_root>/<date>/<name>/` | Recording folder (HDF5, videos, events, resolved settings, config snapshot). `logs/flystate_*.h5` only without `--exp`. |
| `tests/videos/*` | Auto-generated synthetic test fixtures — regenerated by the tests, don't tune configs here. |

### 3.1 `config/rig.yaml`

```yaml
cameras:                      # name -> per-camera YAML (paths relative to this file)
  bottom: camera_bottom.yaml
  front:                      # or {config, enabled} to keep a camera off
    config: camera_front.yaml
    enabled: true
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
| `invertcolor` | `auto` (detect from the first frame), `true` (dark fly on bright IR backlight — the usual bottom view), `false` (bright fly on dark background). Set it explicitly per camera: auto-detection can guess wrong on views without a dark body at the region between the hinges (e.g. a frontal head view). The GUI toggle persists here. |
| `scale_image` | Downscale factor applied to frames before tracking (0.5 halves resolution, ~4x faster). |
| `n_edges_max` | Max edges reported per wing by the edge tracker (1 = leading edge only). |
| `rc_background` | Time constant (s) of the moving-average background model. |
| `wingbeat_min/max` | Expected wingbeat frequency band (Hz) for the aux detector (it uses aliased spectra, so it works at 90 fps). |

**Per part** (`head:`, `abdomen:`, `left:`, `right:`, `aux:`)

| Key | Meaning |
|---|---|
| `tracker` | Algorithm: `edge` (radial edge angle — wings, or antennae for head yaw), `area` (rotation via polar image registration — head), `rotation` (rotation via template rotation search about the hinge — small angles such as head roll from the front; ~1 ms/frame), `tip` (farthest point — abdomen), `intensity` (mean intensity + wingbeat frequency — aux). |
| `autozero` | Auto-center the reported angle on the resting position. |
| `threshold` | Edge-strength threshold (edge tracker); 0 = accept all. |
| `feathering` | Softening of mask edges (0–1). |
| `search_deg` | `rotation` tracker only: half-range of the rotation search in degrees (default 8). |
| `polar_amplify` | Angular resolution multiplier of the polar transform (default 1.0). Raise (2–4) to resolve small rotations (e.g. head roll of a few degrees) at the cost of ~1–3 ms/frame for that part. |
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

One row per merged state: `/merged/seq`, `/merged/t_host`, `/merged/t_wall`, and per camera
`/<cam>/seq`, `t_camera`, `t_host`, `t_done`, `/<cam>/<part>_angle` and
`_radius` for head/abdomen/left/right, `/<cam>/aux_intensity`, `aux_freq`.
Angles are radians; times are seconds (host clock is monotonic; `t_camera` is
the device clock; `t_wall` is `time.time()`). Events: `/events/t_host`, `/events/t_wall`,
`/events/text`. Readable with h5py or MATLAB's `h5read`.

---

## 4. Adapting to a different rig

- **Different cameras:** change only the `camera:` sections. FLIR → set
  serials; other GenICam/USB vendors → use `kind: opencv`, or add a new
  `FrameSource` subclass in `kinefly2/camera/` (implement `open/read/close`
  returning `Frame` objects) and register it in `camera/base.py::make_source`.
- **One camera instead of two:** set `enabled: false` on the camera in
  `rig.yaml`, or run `run_rig ... --cameras bottom`; each camera is an
  independent system with its own config, background and reference files.
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
kinefly2/apps/       run_offline, run_rig, check_latency, dashboard
kinefly2/gui/autoplace.py  automatic region placement (registration / silhouette)
config/              rig.yaml + per-camera YAMLs + saved backgrounds/references
experiments/         per-recording settings templates
tests/               synthetic-video generator + ground-truth and architecture tests
```

## 6. Status / roadmap

- [x] Python 3 port of the tracking core, verified against synthetic ground
      truth (wings r=1.000, head r=0.989).
- [x] Dual-camera threaded pipeline, merged state, ZMQ publisher, HDF5 logger.
- [x] Per-experiment recordings (naming convention, raw MJPEG video, stimulus
      events on a shared clock, config snapshots), browser dashboard.
- [x] Auto-placement of regions (`autoplace`), `rotation` tracker for small
      head-roll angles.
- [ ] Live PySpin validation at 90 Hz on the rig (pending the Linux machine
      and FLIR cameras).
- [ ] Reiser LED-panel closed-loop sink (pending the arena). Verify the
      controller generation first: this targets the classic panels controller;
      G4 uses a different protocol.
