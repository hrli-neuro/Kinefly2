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

### 2.2 Real time on the rig: leave it on, record on demand

```bash
python -m kinefly2.apps.run_rig config/rig.yaml
```

Kinefly2 is meant to stay on for hours (experiments and the intervals in
between): the cameras and tracking never stop, the merged fly state is
published continuously on ZMQ `tcp://*:5556`, and one live window per camera
shows the overlays (`--no-gui` for max headroom; `q` or `exit` quits; a
recording in progress is stopped cleanly on exit). Between recordings nothing
is written to disk.

**Recordings** (HDF5 fly state + optional raw video + events) are sessions
that your experiment script starts and stops over the command socket
(`command_port` in rig.yaml, default 5558):

```python
from kinefly2.client import Kinefly2Client        # needs only pyzmq; file can be copied anywhere
k = Kinefly2Client('localhost')                    # or the rig's IP
with k.recording(fly_id='017', protocol='YAMV', notes='stripe fixation'):
    k.event({'trial': 1, 'event': 'stim_on'})      # logged on the recording's clock
    ...                                            # run the trial (later: drive the panels)
    k.event({'trial': 1, 'event': 'stim_off'})
# recording stops when the block ends (also on exceptions)
k.status()                                         # recording?, per-camera fps/frames, last session
k.snapshot()                                       # PNG of each camera's current frame
k.zero()                                           # this posture = zero (after the fly settles)
```

The same works from a terminal: `python -m kinefly2.apps.ctl start --fly 017 --protocol YAMV`,
`... event '{"trial":1}'`, `... status`, `... stop`, `... --host <rig-ip> status`;
and from the GUI: the `record` button toggles a recording (fields from the
template), and a red **REC name m:ss** indicator shows in every camera window.

Each recording gets its own folder named `{date}_{genotype}_{fly_id}_{protocol}`
(section 2.3). The **experiment template** (`--exp`, default
`experiments/default.yaml`) supplies the rig-wide defaults — `data_root`,
`genotype`, video on/off and quality, `events_port` — and every `start`
overrides fields (`fly_id`, `protocol`, `notes`, `record_video`, `genotype`).
`--record-now --fly 017 --protocol YAMV` records from launch until exit
(one-shot use without a script).

Consume the live state from any language / any machine on the network
(`k.states()` in the Python client does the same):

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

Long runs: a live camera that stops delivering frames is reconnected
automatically (status shows `down`/`reconnects`/`last_frame_age_s`); each
session is its own set of files, so a day of experiments is many small
recordings rather than one giant one.

### 2.3 Recording folders and events

Each recording gets a folder named with the lab convention
`{date}_{genotype}_{fly_id}_{protocol}` (suffix `_2`, `_3`… if it exists):

```
<data_root>/<date>/260819_XHCS_XHCS_017_YAMV/
  260819_XHCS_XHCS_017_YAMV_flystate.h5       angles + timestamps (+ /events)
  260819_XHCS_XHCS_017_YAMV_bottom.avi        raw MJPEG video (record_video: true)
  260819_XHCS_XHCS_017_YAMV_bottom_frames.csv frame_id/t_camera/t_host per video frame
  260819_XHCS_XHCS_017_YAMV_front.avi / _front_frames.csv
  260819_XHCS_XHCS_017_YAMV_events.jsonl      stimulus events as received
  260819_XHCS_XHCS_017_YAMV_experiment.yaml   resolved settings
  configs/                                    snapshot of rig.yaml + camera YAMLs used
```

The raw video makes every recording re-analyzable with different regions or
trackers: `run_offline <folder>/configs/camera_bottom.yaml --video <folder>/<name>_bottom.avi --csv out.csv`.
Video frame *k* corresponds to the sidecar row *k*, whose `frame_id` equals
`/<camera>/seq` in the HDF5.

**Events on the same clock.** Besides `k.event(...)` (acknowledged), any
program can push events fire-and-forget to the events port (`events_port` in
the template, default 5557):

```python
import zmq
sock = zmq.Context().socket(zmq.PUSH)
sock.connect('tcp://localhost:5557')
sock.send_string('{"trial": 12, "event": "closed_loop_start", "gain": 1.5}')
```

Events are stamped on arrival with the fly-state clock (`t_host`) and
wall-clock time (`t_wall`) and written to `/events` in the HDF5 and to the
`.jsonl`. Events arriving while no recording is active are counted and dropped.

### 2.4 Live browser dashboard

```bash
python -m kinefly2.apps.dashboard                       # tracker on this machine
python -m kinefly2.apps.dashboard --zmq tcp://<rig-ip>:5556 --port 8000
```

Open `http://<machine>:8000` from any computer on the network for live
traces of L, R, L−R, head yaw/roll and wingbeat frequency. It is a separate
process reading the ZMQ stream, so it adds no load to the tracker, and the
page has no external dependencies (works on an offline rig network).

### 2.5 Front camera: head roll, yaw and pitch from landmarks

From the front, the head's motion under a stimulus combines **roll, yaw and
pitch**. Only roll is an in-plane rotation; the other two are seen end-on and
appear as a *shift* of the head. The hinge-based trackers
(`edge`/`area`/`rotation`) model a pure rotation about a fixed point, so yaw
corrupts their roll estimate and no hinge placement works. The `landmarks`
tracker (default for the front camera's head) separates all three:

| rotation | axis | in the front image | lever arm |
|---|---|---|---|
| **roll** | anterior–posterior | in-plane rotation | none — **exact** |
| **yaw** | dorsoventral | horizontal shift | A–P depth |
| **pitch** | lateral | vertical shift | **the same** A–P depth |

Put the neck joint at the origin with the landmark centroid at
(lateral, depth, height): a small yaw moves the centroid sideways by
`-yaw × depth`, a small pitch moves it vertically by `+pitch × depth`. The
lateral and height offsets drop out of both. **So one constant converts both
shifts into angles** — see §2.5.2.

#### 2.5.1 Placing landmarks

1. Make sure the camera's YAML has `head: tracker: landmarks` (the
   `+lm`/`-lm` buttons appear only then; restart after changing it — the
   startup line `front: trackers head=landmarks ...` confirms it).
   Click `+lm`, then click on a distinctive head feature; repeat for **4–6**
   landmarks spread over the head: the **edges of both eyes**, the **bright
   inter-ocular patch**, the **antenna bases**. Landmarks are draggable
   handles; `-lm` removes the last one.

   Three rules, each enforced with a warning in the status line:

   - **Put them on texture or an edge.** A point inside a uniform blob cannot
     be matched; such a landmark gets a red ring and is dropped.
   - **Do not make the layout symmetric.** Four points on a rectangle map onto
     themselves under a 180° rotation, so the fit cannot tell 0° from 180° and
     flips between them. The tracker checks this and prints
     `AMBIGUOUS LAYOUT`. (This is a real failure that happened here: four
     points 30 px apart, self-symmetric to 3.2 px, sent the roll past 170° in
     2.5% of frames — while the fit *residual* went **down**, because a
     consistent swap is still perfectly rigid. Spread the points unevenly.)
   - **Use 4 or more.** With two, a rigid fit has only one redundant degree of
     freedom, so the residual cannot detect anything. Below four you get
     `only N landmarks`.

   **Avoid specular highlights** (the bright glints on the curved eye/head
   surface under IR illumination): they stay where the surface faces the light
   while the head turns, so they are not attached to the head. Use the eye
   outline, ommatidia/bristle texture, the inter-ocular stripe and the antenna
   bases. `NON-RIGID x px` appears when the set stops moving as one body
   (highlights, legs, proboscis, or a lost match), and `lmN BAD` names the one
   landmark that fits worst over time — it is also circled amber.

   `search_px` is clamped automatically so the matcher can never reach a
   neighbouring landmark (`search 20->9` in the status line means it was
   reduced). This is what prevents the swap described above; you do not need
   to tune it by hand.

2. The posture at the moment you place/move a landmark becomes the zero
   (reference patches are captured then and stored as
   `config/templates_<camera>_head.npz`). Put the hinge on the neck.
   To re-zero later without touching the landmarks — e.g. after mounting
   a new fly, once it has settled — press `zero` (recaptures all landmark
   templates from the current frame; `area`/`rotation` trackers restart
   too), or from a script `k.zero()` / `ctl zero`; a `zeroed` event is
   written to the recording so you know which posture the angles refer to.

3. Each frame every landmark is located by template matching, and a rigid fit
   of the set gives its **rotation** (the roll) and its **translation** (the
   yaw and pitch channels), after removing the centroid shift that a roll
   about the neck produces.

#### 2.5.2 Yaw and pitch are recorded in pixels until calibrated

**Roll is an angle; yaw and pitch are recorded as displacements in pixels.**
Their shared lever arm — the head's anterior–posterior depth from the neck —
points along the optical axis, so a frontal view cannot measure it and any
angle derived from it would be arbitrarily scaled. Roll is unaffected: it is
the fit's rotation and needs no lever arm or rotation centre at all.

To get absolute angles, fit that one constant against a reference that *can*
measure yaw, then apply it to **both** channels:

```bash
jupyter lab notebooks/calibrate_lever.ipynb
```

The notebook uses `kinefly2/analysis/calibrate.py` and takes the reference yaw
from the bottom camera's `bottom/head_angle` (a true in-plane rotation there),
from an external rotation stage, or from spontaneous yaw pooled over long
recordings; `epochs_from_events()` restricts the fit to stimulus trials. The
fit is robust (Huber) and **gated**: it refuses to hand back a constant unless
the reference yaw spans ≥5°, r² ≥ 0.5 and the bootstrap CI is under 20% of the
estimate. A lever arm fitted from a fly that barely moved its head is not a
measurement but noise with units, and it would silently rescale every yaw and
pitch number afterwards. (The 2026-08-19 cam1/cam2 pair is correctly rejected:
bottom-camera yaw std 0.35° against 0.22° of frame-to-frame noise.)

Once known, put it in the head's section:

```yaml
tracker:
  head:
    tracker: landmarks
    lever_px: 92.4      # px per radian, from the notebook
```

`front/head_yaw` and `front/head_pitch` are then logged in radians and the GUI
switches to degrees. Without it those columns stay NaN rather than a guess.
The constant is specific to the fly, the magnification and the landmark set —
re-fit after moving the camera, changing the lens or refocusing. Calibration
is deliberately **post hoc**: the pixel channels are what get recorded, so a
better constant can always be re-applied to old data.

#### 2.5.3 Outputs

`angles = [roll]` (radians), `radii = [dx, dy]` (raw head shift, px),
`gradients = [dx_corr, dy_corr, residual]` (shift with the roll-induced part
removed, px; fit residual, px), and `extra` (roll precision, quality flags,
and the calibrated angles when available). Logged per camera as
`head_angle` (roll), `head_dx`, `head_dy`, **`head_dx_corr` (yaw)**,
**`head_dy_corr` (pitch)**, `head_fit_residual`, `head_score`,
`head_roll_sigma`, and `head_yaw` / `head_pitch` (NaN until `lever_px` is set).
The dashboard plots roll, yaw and pitch separately.

In the camera window the readout sits with the other parts in the bottom-left
corner, wrapping onto a second line when needed:

```
HEAD:   +2.4 deg +-1.0  yaw +2.0 pitch -1.3 px  5/5  q0.88  search 20->9
```

roll in degrees; `+-1.0` the roll uncertainty this landmark layout can deliver
(`residual / (spread × √N)`, so more and wider-spread landmarks tighten it);
the yaw and pitch channels (px, or degrees once calibrated); how many
landmarks matched; and `q`, the mean template match score (normalized
cross-correlation, 0–1; landmarks below `min_score` are dropped). Warnings
follow.

Parameters (`tracker: head:`): `template_px`, `search_px` (auto-clamped),
`min_score`, `fit: rigid|similarity`, `rotate_templates`, `smooth`,
`nonrigid_px`, `lever_px`. Cost ≈ 0.3–0.6 ms. Focus and magnification of the
front camera matter more than frame rate.

### 2.6 Timing validation

```bash
python -m kinefly2.apps.check_latency config/camera_bottom.yaml --seconds 30
```

Healthy rig numbers at 90 Hz: interval median ≈ 11.1 ms, dropped frames = 0,
track median in the low milliseconds.

### 2.7 GUI controls (same UX as Kinefly)

- **Handles (drag with mouse):** head/abdomen hinge points define the body
  axis; wing hinge points sit on the wing bases; for each part, the two line
  handles set the angular range (`angle_lo`/`angle_hi`) and the two circle
  handles set `radius_inner`/`radius_outer`; the aux ellipse has center and
  two radius handles.
- **Toolbar:** `exit`; `saveBG` (capture the current image as the background
  for subtraction — do it with the fly absent or not illuminated);
  `autoplace` (place all regions automatically from the current image, see
  below); `reset` (undo autoplace / restore the startup layout); `zero` (this
  posture = zero for all trackers: landmark templates recaptured);
  `track: H A L R X` (toggle head/abdomen/left/right/aux);
  `subtract: H A LR X` (background subtraction per part); `invertcolor`
  (set if the fly is dark on a bright background); `stabilize`; `symmetric`
  (mirror left/right edits); `windows` (extra debug windows).
- **Every mouse-up saves the GUI-owned parameters (regions, flags) back into
  that camera's YAML**, merged into the file as it is on disk — hand edits
  made while Kinefly2 runs (tracker types, thresholds) are kept, with a
  console note when a tracker type changed on disk (restart to apply) — and
  stores the current frame as `config/reference_<camera>.png`, the frame the
  regions were tuned on.

**Auto-placement (`autoplace`) and `reset`.** For a new fly, click
`autoplace`: the current frame is registered to the saved reference frame
(rotation + translation, ECC) and every hinge/center moves accordingly, so a
fly mounted a few pixels or degrees off needs at most a nudge. The status
line reports the shift/rotation found. Safety: the result is applied only if
the transform is plausible (shift < 25% of the image, rotation < 30°) and
keeps every region inside the image — otherwise nothing changes and the
status line says why. Before applying, the camera YAML is backed up to
`<camera>.yaml.bak`. `reset` undoes the last autoplace; pressing it again
(or when there is nothing to undo) restores the layout loaded at startup.
If a config already has tuned regions but no reference frame, the first
frame of the session becomes the reference automatically. Each camera has
its own reference.

---

## 3. Which parameters live in which files

| File | Contents |
|---|---|
| `config/rig.yaml` | Rig topology: camera list, primary camera, ZMQ state port, command port. |
| `config/camera_bottom.yaml` | Bottom view: `camera:` (capture settings) + `tracker:` (all tracking params, GUI-managed). |
| `config/camera_front.yaml` | Same, for the front view. |
| `config/background_<name>.png` | Per-camera background image, written by `saveBG` or `--make-bg`. |
| `config/reference_<name>.png` | Per-camera reference frame for `autoplace` (saved with the params). |
| `experiments/*.yaml` | Per-recording settings: data root, genotype/fly/protocol, notes, video on/off, event port (section 2.3). |
| `kinefly2/gui/app.py` → `tracker_defaults()` | Defaults used for any tracker parameter missing from a camera YAML. |
| `<data_root>/<date>/<name>/` | Recording folder (HDF5, videos, events, resolved settings, config snapshot). |
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
command_port: 5558            # command socket for experiment scripts (kinefly2.client)
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
| `tracker` | Algorithm: `edge` (radial edge angle — wings, or antennae for head yaw), `area` (rotation via polar image registration — head), `rotation` (rotation via template rotation search about the hinge; ~1 ms/frame), `landmarks` (user-placed landmarks + rigid fit → roll, yaw and pitch on separate channels; the front-camera head default, section 2.5), `tip` (farthest point — abdomen), `intensity` (mean intensity + wingbeat frequency — aux). |
| `autozero` | Auto-center the reported angle on the resting position. |
| `threshold` | Edge-strength threshold (edge tracker); 0 = accept all. |
| `feathering` | Softening of mask edges (0–1). |
| `search_deg` | `rotation` tracker only: half-range of the rotation search in degrees (default 8). |
| `template_px`, `search_px`, `min_score`, `fit`, `rotate_templates`, `smooth`, `nonrigid_px`, `lever_px` | `landmarks` tracker only: patch size; search radius (**auto-clamped** so the matcher cannot reach a neighbouring landmark — see 2.5.1); acceptance threshold; `rigid`/`similarity` fit; rotated re-matching for large rolls; output EMA; the fit residual above which the overlay warns `NON-RIGID`; and `lever_px`, the px-per-radian constant that turns the yaw and pitch channels into angles (unset = those columns stay NaN; fit it with `notebooks/calibrate_lever.ipynb`, see 2.5.2). Landmark positions live in `gui.<part>.landmarks`. |
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
With the `landmarks` tracker on the head, also `/<cam>/head_dx`, `head_dy`
(raw head translation, px), **`head_dx_corr`** and **`head_dy_corr`** (roll-induced
part removed — from a front view these are the head's **yaw** and **pitch**,
seen end-on, both in px through the same lever arm), `head_fit_residual`,
`head_score`, `head_roll_sigma` (roll uncertainty, rad), and `head_yaw` /
`head_pitch` (rad — NaN unless `lever_px` is calibrated, section 2.5.2). All
NaN for other trackers.
Times are seconds (host clock is monotonic; `t_camera` is the device clock;
`t_wall` is `time.time()`). Events: `/events/t_host`, `/events/t_wall`,
`/events/text`. Readable with h5py or MATLAB's `h5read`.

**Units.** Angles are **radians** everywhere they are stored or streamed
(HDF5, ZMQ, the CSV from `run_offline`), as in the original Kinefly. The GUI
overlay and the dashboard **display degrees** (`HEAD: 2.4 deg`), and
displacements are always pixels.

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
  and its `head` `landmarks` tracker for head roll, yaw and pitch (section 2.5). Head yaw from below can use
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
kinefly2/apps/       run_offline, run_rig, ctl (command-line control), check_latency, dashboard
kinefly2/client.py   Python client for experiment scripts (start/stop/event/status/states)
kinefly2/core/session.py   recording sessions (attach/detach logger + video at runtime)
kinefly2/gui/autoplace.py  automatic region placement (registration / silhouette)
kinefly2/analysis/   offline analysis: calibrate.py (yaw/pitch lever arm) - not in the real-time path
config/              rig.yaml + per-camera YAMLs + saved backgrounds/references
experiments/         per-recording settings templates
notebooks/           calibrate_lever.ipynb - post-hoc yaw/pitch calibration
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
- [x] Continuous operation with script-triggered recording sessions (command
      socket + Python client + CLI + GUI record button), camera reconnect.
- [x] Front-camera head roll + yaw via landmark tracking (`landmarks` tracker).
- [x] Head **pitch** as the landmark tracker's vertical channel, sharing one
      lever arm with yaw; landmark-layout safety checks (ambiguous layouts,
      auto-clamped search radius, per-landmark residuals, roll precision).
- [ ] Fit `lever_px` and turn the yaw/pitch pixels into angles - needs a
      recording with real head yaw (optomotor stimulus, or a rotation stage);
      `notebooks/calibrate_lever.ipynb` is ready and currently rejects the
      2026-08-19 pair for having only 0.35 deg of yaw.
- [ ] Live PySpin validation at 90 Hz on the rig (pending the Linux machine
      and FLIR cameras).
- [ ] Reiser LED-panel closed-loop sink (pending the arena). Verify the
      controller generation first: this targets the classic panels controller;
      G4 uses a different protocol.
