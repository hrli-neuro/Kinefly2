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

### 2.1.1 Connecting the cameras

Everything above runs from video files. To run from real cameras, the **only**
thing Kinefly2 needs to identify one is its **serial number** — that is what
binds a physical camera to `bottom` or `front` regardless of which port it is
in or what order the two enumerate. Never rely on index order for a two-camera
rig; plug the cameras in the other way round and the views would swap.

Find the serials on the rig:

```bash
python -m kinefly2.apps.list_cameras
```

It prints each camera's serial, model, link speed, resolution and maximum frame
rate at the current ROI, followed by a ready-to-paste `camera:` block. Put each
block into the matching camera YAML in place of the `kind: video` lines (both
shipped configs carry the block already, commented out):

```yaml
camera:
  kind: pyspin
  serial: "22334455"      # binds THIS camera to 'bottom' whatever port it is in
  fps: 90
  exposure_us: 2000       # must be < 1/fps; set it manually
  gain_db: 0
  roi: {width: 512, height: 512}    # centred unless offset_x/offset_y are given
  trigger: {mode: primary}          # the other camera gets {mode: secondary}
```

Nothing else changes. `make_source` swaps the frame source and the tracking,
GUI, recording and ZMQ paths never know the difference — which is also why the
whole system can be developed against video files.

Two settings matter more than they look:

- **`exposure_us`** must be shorter than the frame interval (11111 µs at 90 Hz),
  and must be set *manually*. Auto-exposure varies the exposure time frame to
  frame, which changes apparent brightness and blur and so moves the tracked
  edges.
- **`roi`** is usually what actually buys the frame rate: less to move over the
  bus. Crop to the fly and check the "max fps" line from `list_cameras`.
- **`trigger`** puts both cameras on one frame clock — one `primary` strobing
  its exposure on an output line, the other `secondary` following it. Free-
  running cameras drift, so a frame from one can sit up to a frame away from
  the other's. That matters here specifically because the two views are
  compared directly: the front camera's lever arm is calibrated against the
  bottom camera's angle (section 2.3.1), and the r = 0.97 that calibration
  achieved came from two views that *were* frame-synchronised. Wire the
  primary's strobe line to the secondary's trigger input per FLIR's GPIO
  pinout for your model.

> The camera code paths cannot be exercised without the hardware, so ROI,
> binning and trigger are implemented but **untested on a real camera**. The
> config plumbing is covered by tests; the rest wants a careful first run with
> `list_cameras` and a short recording before trusting a session to it.

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

#### 2.3.1 Per-fly calibration: `lever_px`

The front camera's pitch and yaw channels are pixels until you supply one
constant per fly. It depends on head size, mounting **and landmark placement**,
so it lives in the **experiment YAML**, never in the camera config — a constant
inherited from the previous fly would silently rescale every angle:

```yaml
lever_px:
  front: 48.2      # px per radian, for THIS fly
```

Fit it from the fly's **own recording** — no stimulus needed. The reference is
the bottom camera's exact yaw, and spontaneous head movement is usually plenty
(the reference recording calibrated at r² = 0.96 from 32° of spontaneous yaw):

```bash
python -m kinefly2.apps.calibrate_fly <recording_dir> --write experiments/today.yaml
```

Calibrating from the fly's own recording — rather than a separate calibration
block — guarantees the constant belongs to the landmark set the data was taken
with. Nothing is lost by doing it afterwards: the pixel channels are always
logged, so the first recording can be converted retrospectively and later
blocks of the same fly get degrees live.

`calibrate_fly` refuses to hand back a constant unless the fit passes its gates
(reference yaw spanning ≥5°, r² ≥ 0.5, bootstrap CI under 20%), and it checks
two things the numbers silently depend on:

- **the landmarks**, compared against the camera config's current ones. Moving
  one shifts the centroid and so the lever arm; a constant from before the move
  does not describe the current setup, and `--write` is refused (`--force`
  overrides).
- **the axis** (§2.5.2), fitted from the data. If the config's `axis_deg` is
  off the fitted line, pitch and yaw are probably swapped and it says so.

At recording start the constant is applied to the trackers for that recording,
so the GUI shows degrees and the log carries angles; it is stamped into
`<run>_experiment.yaml` and into the HDF5 attributes together with the landmark
positions, so a recording always says what its angles were derived from.

Typical loop: mount fly → place landmarks → record → `calibrate_fly --write` →
subsequent blocks report degrees.

### 2.4 Live browser dashboard

```bash
python -m kinefly2.apps.dashboard                       # tracker on this machine
python -m kinefly2.apps.dashboard --zmq tcp://<rig-ip>:5556 --port 8000
```

Open `http://<machine>:8000` from any computer on the network. It is a separate
process reading the ZMQ stream, so it adds no load to the tracker, and the page
has no external dependencies (it works on an offline rig network).

It behaves like a chart recorder rather than an auto-ranging plot: **each trace
holds a fixed y range**, so the picture stays still and a value now is
comparable with the same value a minute ago. Autoscaling every redraw hides
small changes by normalising them away.

| control | what it does |
|---|---|
| `window` | Visible span — 5 / 10 / 30 / 60 s, always ending at now. |
| `fit` (per trace) | Set this trace's range from what is currently on screen, once. |
| `+` `−` `▲` `▼` | Zoom and pan that trace's y axis. |
| min/max boxes | Type an exact range. |
| `×` | Hide the trace; it reappears as a button below the plots. |
| drag the title | Reorder panels. |
| `Pause` | Freeze the display without losing the stream; `Live` resumes. |
| hover | Cursor line with every trace's value at that instant. |
| `Fit all` / `Reset layout` | Autofit everything, or go back to defaults. |

Ranges, order and hidden traces are remembered in the browser, so a reload
brings back your working setup. Panels with no data are hidden automatically
on first connect, so running only the bottom camera does not leave half the
page empty.

Traces are drawn with **min/max decimation** — one vertical segment per pixel
column — so a 60 s window at 90 Hz (5400 points) draws quickly and shows the
true envelope instead of an aliased subsample. A brief spike is never skipped.

Channel names come from the tracker's own `extra`, so they mean what they say
whatever the camera view: a ventral camera shows only its exact yaw, and a
frontal camera shows roll plus pitch and yaw — in **degrees** once that fly's
`lever_px` is set (section 2.3.1) and in **pixels** before that. Reading the
raw image-axis displacement instead would mislabel both channels on any camera
whose axis is not aligned with the frame.

**Stimulus row.** `/merged/stim` is drawn as a step trace. Set it from the
experiment script whenever the pattern changes:

```python
k.stim(2.5)      # the voltage code now driving the panels
k.stim(0)        # off
```

The value is **held** until it changes, so every frame is labelled with the
stimulus that was on, and `/merged/stim` in the recording answers "which
stimulus was on at frame N" as a direct lookup. `/merged/stim_t_wall` records
when it last changed, so update latency stays visible. `StateMerger.set_stim`
is the seam an analog input would use — a DAQ reader calls exactly that, and
nothing downstream can tell the difference.

### 2.5 Head tracking with landmarks (both cameras)

The `landmarks` tracker is the default for the head on **both** cameras. It
places a few points on distinctive head features, matches each one by template
matching every frame, and fits a rigid transform to the set. That fit gives one
rotation **exactly** — the in-plane one, which needs no rotation centre and no
scale, so it does not matter where the hinge sits.

**Which rotation that is depends on the view**, set by `view:` at the top of
the camera's `tracker:` section:

| `view` | camera | exact | along the part's axis | across it |
|---|---|---|---|---|
| `ventral` | below the fly | **yaw** | — | — |
| `frontal` | in front | **roll** | **pitch** | **yaw** |

Between them the two cameras measure **yaw and roll exactly, and pitch once**.

**A ventral camera reports yaw only.** Its lever arm is the head's
dorsoventral depth from the neck, far shorter than the anterior–posterior depth
a frontal camera gets, and the two displacement channels measured noise: SNR
1.9 and 1.2 against a 0.28 px centroid noise floor, and the "roll" channel
against the front camera's *exact* roll gave r = −0.15 — it was not measuring
roll at all. So they are not reported. (`VIEW_CHANNELS` in
`kinefly2/tracking/tracker_landmarks.py` records this; re-enabling is one edit.)

Why not the hinge-based trackers (`edge`/`area`/`rotation`): they model a pure
rotation about a fixed point, so any translation of the head corrupts the
angle. On `cam1.mp4` the bottom head gave SNR 2.26 with `area` against **5.48**
with landmarks, and a 0.56 px fit residual confirming the points move as one
rigid body. The two disagree (r = −0.13), and `area` also shifts with its
annulus — re-aiming the mask moved its yaw std from 0.35° to 1.83°.

The wings keep `edge` tracking and the abdomen keeps its own tracker: the
abdomen is measurably **not rigid** (2.57 px residual against the head's 0.41,
SNR 1.28, and a similarity fit is worse still at 2.98 px), so a rigid landmark
fit is the wrong model for it.

#### 2.5.1 Placing landmarks

`+H` / `-H` add and remove landmarks on the head (one button pair per landmark
part). Click `+H`, then click the feature. Aim for **4–6** points spread over
the head. Landmarks are draggable handles.

Where they go depends on the view:

- **Ventral (bottom camera):** *anterior of the neck hinge only.* Points behind
  it land on the thorax and forelegs, which move independently — in testing
  that turned a clean 9.6° yaw signal into 13.6° of garbage.
- **Frontal (front camera):** the edges of both eyes, the bright inter-ocular
  patch, the antenna bases.

Four rules, each enforced with a warning on the status line:

- **Put them on texture or an edge.** A point inside a uniform blob cannot be
  matched; it gets a red ring and is dropped.
- **Keep them apart** — at least `template_px`, ideally more. `search_px` is
  clamped automatically so the match window
  (`search_px + template_px//2`) can never reach a neighbouring landmark and
  steal its feature. `search 10->3 (landmarks 11px apart)` means the clamp got
  severe and the points are crowded; spread them and it relaxes.
- **Do not make the layout symmetric.** Four points on a rectangle map onto
  themselves under a 180° rotation, so the fit cannot tell 0° from 180° and
  flips. You get `AMBIGUOUS LAYOUT`. (A real case: four points 30 px apart,
  self-symmetric to 3.2 px, sent the roll past 170° in 2.5% of frames — while
  the fit *residual* went **down**, because a consistent swap is still
  perfectly rigid.)
- **Use 4 or more.** With two, a rigid fit has one redundant degree of freedom,
  so the residual cannot detect anything. Below four you get `only N landmarks`.

**Avoid specular highlights** — the IR glints on the curved eye surface stay
where the surface faces the light while the head turns, so they are not
attached to the head. `NON-RIGID x px` appears when the set stops moving as one
body, and `lmN BAD` names the single worst-fitting point (also circled amber).

**What actually sets the precision.** Matching is already sub-pixel: a 3-point
parabola is fitted to the correlation peak in x and y, and the matched
positions land uniformly across the pixel (0.071–0.124 against an ideal 0.100),
so there is no pixel-locking. But it is not the limiting factor — turning it
off costs only 12% on the bottom camera and 1% on the front. A more elaborate
interpolator would buy correspondingly little. What does move the number:

| lever | measured effect |
|---|---|
| **placement** | per-landmark residuals span 0.61 → 1.79 px across one 5-point set |
| **`template_px`** | 25 → 33 on the front camera: residual 1.58 → 1.21 px (23%) |
| rotation | 1.19 px at \|roll\| < 3° against 1.73 px at \|roll\| > 10° |

So: watch `lmN BAD` and move the offender, and size `template_px` to the head
rather than leaving the default. A bigger template needs proportionally more
spacing, because `search_px` is clamped against the gap between landmarks —
crowd them and you get `search 10->3 (landmarks 11px apart)` and lose the room
to follow fast movement. Spread first, then enlarge.

The posture when you place or move a landmark becomes the zero. Press `zero`
later to re-zero without touching the landmarks; a `zeroed` event is written to
the recording.

Parts that are switched off draw **nothing** — not even their hinge — so a
camera tracking only the head shows only the head. Toggle a part on in the
toolbar to bring its handles back.

#### 2.5.2 The axis handle (frontal cameras)

Pitch is the displacement **along** one image direction and yaw the component
**across** it. That needs a direction, and image x/y will not do — it depends
on how the camera is mounted. So the frontal camera's head carries an
`axis_deg`, drawn as a draggable arrow from the hinge labelled `+pitch`.

**Point it along the fly's dorsoventral axis in the image** — not along the
head's widest dimension, which is left–right in a frontal view. If the camera
is mounted rotated, the dorsoventral axis may run horizontally in the frame;
on the reference rig it does (`axis_deg` ≈ 173°). Getting it wrong silently
swaps pitch and yaw, so don't guess: `calibrate_fly` (below) fits the axis from
data and tells you if the config disagrees.

Only the direction matters; the arrow's length is cosmetic. The arrow also
fixes the sign: positive pitch means the landmarks move along it. A ventral
camera has no axis handle, because it reports no displacement channels.

#### 2.5.3 Pitch and yaw are in pixels until the fly is calibrated

The frontal camera's two displacement channels share one lever arm: the depth
of the landmark centroid from the neck along the optical axis. That depth
cannot be measured from the view itself, so the channels are recorded in
**pixels**. The exact in-plane angle is unaffected.

The constant depends on head size, mounting **and landmark placement**, so it
is **per fly** — see §2.3.1.

#### 2.5.4 Outputs

Per camera: `head_angle` (the in-plane angle, radians), `head_dx`, `head_dy`
(raw shift, px), `head_dx_corr`, `head_dy_corr` (image axes, rotation removed),
`head_d_axial` and `head_d_lateral` (along and across the axis, px — NaN on a
ventral camera), `head_fit_residual`, `head_score`, `head_roll_sigma`, and
**`head_roll`, `head_pitch`, `head_yaw`** — the three rotations under fixed
anatomical names, so analysis never needs to know the view. Exactly one is
always finite (the exact in-plane angle); on a frontal camera the other two
fill in once `lever_px` is known, on a ventral camera they never do.

In the camera window the readout sits bottom-left, wrapping when needed:

```
HEAD: yaw    +2.4 deg +-1.0  4/4  q0.98                       (ventral)
HEAD: roll   +3.0 deg +-0.8  pitch  +2.4 yaw +10.5 deg  6/6  q0.86   (frontal, calibrated)
```

The exact channel in degrees with `+-` its precision (`residual / (spread ×
√N)` — more and wider-spread landmarks tighten it), then, on a frontal camera,
the two lever-arm channels in px (degrees once calibrated), how many landmarks
matched, and `q`, the mean template match score (0–1). Warnings follow.

Parameters (`tracker: head:`): `template_px`, `search_px` (auto-clamped),
`min_score`, `fit`, `rotate_templates`, `smooth`, `nonrigid_px`; geometry under
`gui.head:` is `hinge`, `axis_deg` (frontal only), `landmarks`. Cost ≈
0.3–0.6 ms.

### 2.6 Timing validation

```bash
python -m kinefly2.apps.check_latency config/camera_bottom.yaml --seconds 30
```

Healthy rig numbers at 90 Hz: interval median ≈ 11.1 ms, dropped frames = 0,
track median in the low milliseconds.

### 2.7 GUI controls (same UX as Kinefly)

- **Handles (drag with mouse):** hinge points mark each part's joint; wing
  hinge points sit on the wing bases; for each part, the two line handles set
  the angular range (`angle_lo`/`angle_hi`) and the two circle handles set
  `radius_inner`/`radius_outer`; the aux ellipse has center and two radius
  handles. `landmarks` parts add draggable landmark points and an **axis
  arrow** (labelled `+pitch`) setting the part's long axis — section 2.5.2.
  Hinge moves no longer disturb the wing angles: the body frame they are
  reported in is frozen in `body_axis_deg`.
- **Toolbar:** `exit`; `saveBG` (capture the current image as the background
  for subtraction — do it with the fly absent or not illuminated);
  `autoplace` (place all regions automatically from the current image, see
  below); `reset` (undo autoplace / restore the startup layout); `zero` (this
  posture = zero for all trackers: landmark templates recaptured);
  `+H`/`-H` (add/remove a landmark on the head — one pair per landmark part);
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
| `config/camera_bottom.yaml` | Bottom view: `camera:` (capture settings — video file or live camera, section 2.1.1) + `tracker:` (all tracking params, GUI-managed). Only the `tracker:` block is rewritten by the GUI, so comments in `camera:` survive. |
| `config/camera_front.yaml` | Same, for the front view. Note it holds no `lever_px`: that is per fly, not per rig. |
| `config/background_<name>.png` | Per-camera background image, written by `saveBG` or `--make-bg`. |
| `config/reference_<name>.png` | Per-camera reference frame for `autoplace` (saved with the params). |
| `experiments/*.yaml` | Per-recording settings: data root, genotype/fly/protocol, notes, video on/off, event port (section 2.3), and **`lever_px`, the per-fly head calibration** (section 2.3.1). |
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
| `view` | `ventral` (camera below the fly) or `frontal` (in front). Decides which rotation the landmark tracker measures exactly, and whether it reports displacement channels at all — section 2.5. |
| `body_axis_deg` | The body frame the **wing** angles are reported in. Seeded once from the head/abdomen hinges, then frozen, so re-placing a hinge cannot silently shift every wing angle. Delete the key to re-seed it. |

**Per part** (`head:`, `abdomen:`, `left:`, `right:`, `aux:`)

| Key | Meaning |
|---|---|
| `tracker` | Algorithm: `edge` (radial edge angle — wings, or antennae for head yaw), `area` (rotation via polar image registration — head), `rotation` (rotation via template rotation search about the hinge; ~1 ms/frame), `landmarks` (user-placed landmarks + rigid fit → one exact rotation plus two lever-arm channels; the head default on **both** cameras, section 2.5), `tip` (farthest point — abdomen), `intensity` (mean intensity + wingbeat frequency — aux). |
| `autozero` | Auto-center the reported angle on the resting position. |
| `threshold` | Edge-strength threshold (edge tracker); 0 = accept all. |
| `feathering` | Softening of mask edges (0–1). |
| `search_deg` | `rotation` tracker only: half-range of the rotation search in degrees (default 8). |
| `template_px`, `search_px`, `min_score`, `fit`, `rotate_templates`, `smooth`, `nonrigid_px` | `landmarks` tracker only: patch size; search radius (**auto-clamped** so the matcher cannot reach a neighbouring landmark — see 2.5.1); acceptance threshold; `rigid`/`similarity` fit; rotated re-matching for large rolls; output EMA; and the fit residual above which the overlay warns `NON-RIGID`. Landmark positions and the axis live in `gui.<part>.landmarks` / `gui.<part>.axis_deg`. The px-per-radian constant `lever_px` is **not** here: it is per fly and lives in the experiment YAML (section 2.3.1). |
| `polar_amplify` | Angular resolution multiplier of the polar transform (default 1.0). Raise (2–4) to resolve small rotations (e.g. head roll of a few degrees) at the cost of ~1–3 ms/frame for that part. |
| `saturation_correction` | Compensate saturated pixels when background-subtracting. |

**Geometry, per part under `gui:`** — what the handles edit

| Key | Meaning |
|---|---|
| `track` | Whether this part is tracked (toolbar checkbox). |
| `subtract_bg` | Background subtraction for this part (needs a saved background). |
| `hinge: {x, y}` | Rotation center in image pixels (wing base, neck, abdomen base). The `landmarks` tracker does not need it for its angle, only for the roll-shift correction and as the axis origin. |
| `axis_deg` | Frontal `landmarks` parts: the image direction that counts as +pitch, along the fly's dorsoventral axis. Draggable arrow in the GUI; splits the displacement into pitch (along) and yaw (across) — section 2.5.2. Ventral parts have none. |
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
(raw translation, px), `head_dx_corr` / `head_dy_corr` (rotation-induced part
removed, image axes), **`head_d_axial`** and **`head_d_lateral`** (the same
shift along and across the part's own axis, px), `head_fit_residual`,
`head_score`, `head_roll_sigma` (rad), and **`head_roll`, `head_pitch`,
`head_yaw`** (rad) — the three rotations under fixed anatomical names, so
analysis need not know the view. Exactly one of those three is always finite
(the exact in-plane angle: yaw for a ventral camera, roll for a frontal one);
on a frontal camera the other two fill in once `lever_px` is known (section
2.3.1); on a ventral camera they stay NaN by design, as do its `head_d_axial`
and `head_d_lateral` — the column set is kept uniform across cameras. All NaN
for other trackers.
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
kinefly2/apps/       run_offline, run_rig, ctl, list_cameras, calibrate_fly, check_latency, dashboard
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
- [x] Head **pitch** as the landmark tracker's axial channel, sharing one lever
      arm with the lateral one; landmark-layout safety checks (ambiguous
      layouts, auto-clamped search radius, per-landmark residuals, precision).
- [x] Landmark head tracking on **both** cameras with per-view channel naming
      (ventral -> yaw exact, frontal -> roll exact) and a per-part axis handle;
      wing angles decoupled from the head/abdomen hinges (`body_axis_deg`).
- [x] Ventral view reports yaw only - its pitch/roll channels measured noise
      (SNR 1.9 and 1.2). Untracked parts draw nothing.
- [x] Per-fly `lever_px` in the experiment YAML, fitted from the fly's own
      recording by `calibrate_fly` with landmark-staleness and axis checks.
      Head **pitch in degrees** on the front camera (r2 0.96; yaw cross-checks
      against the bottom camera at r = 0.97).
- [x] Camera hookup: `list_cameras` prints each camera's serial and a pasteable
      config block; ROI, binning and hardware trigger in the camera YAML.
- [x] Stimulus code channel (`/merged/stim`), held per frame, drawn live.
- [x] Chart-recorder dashboard: fixed per-trace ranges, adjustable window,
      min/max decimation, pause with cursor, reorder/hide, saved layout.
- [ ] First run against real FLIR cameras: verify ROI and the primary/secondary
      trigger wiring, and confirm 90 Hz on both views at the chosen ROI.
- [ ] Confirm the lever arm across several flies, and decide whether it can be
      normalised by a size proxy (eye-to-eye distance in px) instead of being
      re-fitted per fly.
- [ ] Live PySpin validation at 90 Hz on the rig (pending the Linux machine
      and FLIR cameras).
- [ ] Reiser LED-panel closed-loop sink (pending the arena). Verify the
      controller generation first: this targets the classic panels controller;
      G4 uses a different protocol.
