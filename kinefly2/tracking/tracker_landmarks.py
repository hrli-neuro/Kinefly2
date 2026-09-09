"""LandmarkTracker: one exact rotation angle plus two displacement channels,
from user-placed landmarks. Used for the head in both camera views.

The user places a few landmarks on distinctive features (eye edges, the bright
inter-ocular patch, antenna bases). Each frame every landmark is located by
normalized template matching of its reference patch inside a small search
window; a rigid (or similarity) fit of the reference positions onto the
measured ones decomposes the motion:

    rotation of the fit    -> the IN-PLANE angle. Exact: it needs no rotation
                              centre, so it does not matter where the hinge is.
    translation of the fit -> the DISPLACEMENT in pixels, after removing the
                              centroid shift that an in-plane rotation about
                              the hinge produces.

Which anatomical rotation each channel carries depends on the VIEW. The two
displacement components are the two rotations seen end-on, decomposed along the
part's own long axis (`axis_deg`, pointing away from its joint) rather than
along image x/y -- the fly is rarely square to the frame:

    view: ventral (camera below the fly)
        in-plane angle -> YAW    (about the dorsoventral axis)  EXACT
        along the axis -> PITCH  (about the lateral axis)
        perpendicular  -> ROLL   (about the anterior-posterior axis)

    view: frontal (camera in front of the fly)
        in-plane angle -> ROLL   (about the anterior-posterior axis)  EXACT
        along the axis -> PITCH  (about the lateral axis)
        perpendicular  -> YAW    (about the dorsoventral axis)

Pitch is the axial channel in both views. Each view measures ONE rotation
exactly -- the in-plane one, which needs no rotation centre and no scale -- and
the other two only through a lever arm: the depth of the landmark centroid from
the joint along the optical axis. That depth cannot be measured from the view
itself, so those two channels stay in PIXELS. One constant converts both; fit
it post hoc with kinefly2.analysis.calibrate against the OTHER camera's exact
in-plane angle. Set it as `lever_px` in the part's config section and the
angles appear in the log and the GUI; leave it unset and they are NaN rather
than a guess.

Sign: positive pitch = the landmarks move along +axis (away from the joint);
positive roll/yaw = they move along +perpendicular, which is +axis rotated 90
degrees clockwise on screen. Which physical direction that is depends on which
side of the joint the landmarks sit, so check it once against a known head
movement and add 180 to `axis_deg` if the sign is inverted.

Unlike the hinge-based trackers (edge/area/rotation) this needs no rotation
center and is not corrupted by combined roll+yaw. Blob-like features also
survive soft focus. Cost: N small matchTemplate calls, well under 1 ms.

Outputs (like the other trackers: angles in radians; the GUI displays degrees):
    state.angles    = [roll]                        radians
    state.radii     = [dx, dy]                      raw centroid shift, px
    state.gradients = [dx_corr, dy_corr, residual]  roll-removed shift, px; fit residual, px
    state.intensity = mean match score              0..1 quality
    state.kind      = 'landmarks'
    state.extra     = {roll, pitch, yaw (rad; only the in-plane one is finite
                       until lever_px is set), d_axial, d_lateral (px),
                       roll_sigma, n_ok, worst_lm, warnings}
Camera-level parameter (tracker: section): view ('ventral'|'frontal').
Per-part parameters (tracker: section): template_px, search_px, min_score,
fit ('rigid'|'similarity'), rotate_templates, smooth, nonrigid_px (residual
above which the overlay warns that the landmarks do not move as one body:
landmarks on specular highlights, legs, or the proboscis, or a mismatch),
lever_px (optional; see above).
Geometry (gui.<part>): hinge (the joint), axis_deg (long axis, draggable
in the GUI), landmarks: [{x, y}, ...].
"""
import os

import cv2
import numpy as np

from . import ui
from .MsgFlyState import MsgState


def rigid_fit(P, Q, similarity=False):
    """Least-squares transform mapping points P (Nx2) onto Q (Nx2):
    Q ~ s*R*P + t. Returns (theta, scale, dcentroid, residuals)."""
    Pc, Qc = P.mean(0), Q.mean(0)
    P0, Q0 = P - Pc, Q - Qc
    H = P0.T @ Q0
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, d])
    R = Vt.T @ D @ U.T
    theta = float(np.arctan2(R[1, 0], R[0, 0]))
    scale = float((S * np.diag(D)).sum() / max(1e-9, (P0 ** 2).sum())) if similarity else 1.0
    resid = np.linalg.norm(Q0 - (scale * (R @ P0.T)).T, axis=1)
    return theta, scale, Qc - Pc, resid


def layout_metrics(pts):
    """Properties of a landmark layout that decide what it can measure.

    rms_radius  spread about the centroid; roll precision scales with it.
    min_dist    closest pair; bounds how far the matcher may search (below).
    ambiguity   how well the pattern maps onto itself when rotated. A layout
                that reproduces itself under a rotation cannot be told apart
                from that rotation, so the fit flips between the two.
    """
    P = np.asarray(pts, dtype=float).reshape(-1, 2)
    n = len(P)
    if n < 2:
        return {'n': n, 'rms_radius': 0.0, 'min_dist': np.inf, 'ambiguity': np.inf}
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    min_dist = float(d[~np.eye(n, dtype=bool)].min())
    c = P.mean(0)
    rms = float(np.sqrt((np.linalg.norm(P - c, axis=1) ** 2).mean()))
    amb = np.inf
    for deg in (180.0, 90.0, -90.0):
        a = np.radians(deg)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        Pr = (R @ (P - c).T).T + c
        # Worst case over points: how far the rotated pattern sits from the
        # original. Small = the rotation reproduces the layout = ambiguous.
        amb = min(amb, float(np.linalg.norm(Pr[:, None, :] - P[None, :, :],
                                            axis=2).min(1).max()))
    return {'n': n, 'rms_radius': rms, 'min_dist': min_dist, 'ambiguity': amb}


def safe_search_px(min_dist, template_px):
    """Largest search radius that cannot reach a neighbouring landmark.

    matchTemplate scans out to search_px + template_px//2 from the predicted
    position. Once that reaches the next landmark, a point can lock onto its
    neighbour's feature; on a near-symmetric layout the whole set swaps at once
    and the fit reports a spurious rotation while staying perfectly rigid, so
    the residual check cannot see it. Measured case: 4 landmarks 30.0 px apart
    with search_px 20 and template_px 25 reached 32 px and flipped the roll
    past 170 deg in 2.5% of frames; clamping the reach removed all of them.
    The 0.9 factor leaves room for the prediction itself being slightly off.
    """
    if not np.isfinite(min_dist):
        return None
    return max(2, int(0.9 * min_dist) - template_px // 2)


# Which rotation each of the three channels carries, per camera view.
# 'inplane' is the exact one; 'axial' and 'lateral' need the lever arm.
VIEW_CHANNELS = {
    'ventral': {'inplane': 'yaw',  'axial': 'pitch', 'lateral': 'roll'},
    'frontal': {'inplane': 'roll', 'axial': 'pitch', 'lateral': 'yaw'},
}
MIN_AXIS_PX = 8.0      # hinge->centroid shorter than this cannot define an axis


def axis_vectors(axis_deg):
    """Unit vector along the part's long axis and its perpendicular.

    The perpendicular is the axis rotated +90 deg, which in image coordinates
    (y down) is clockwise on screen. Returns (axial, lateral)."""
    a = np.radians(float(axis_deg))
    ax = np.array([np.cos(a), np.sin(a)])
    return ax, np.array([-ax[1], ax[0]])


def match_landmarks(image, templates, predicted, search_px, margin):
    """Locate each template near its predicted position. Returns
    (positions Nx2, scores N); positions are NaN where matching failed."""
    H, W = image.shape[:2]
    n = len(templates)
    pos = np.full((n, 2), np.nan)
    scores = np.zeros(n)
    for k, tpl in enumerate(templates):
        if tpl is None:
            continue
        th, tw = tpl.shape
        hh, hw = th // 2, tw // 2
        px, py = predicted[k]
        if not (np.isfinite(px) and np.isfinite(py)):
            continue
        x0 = int(round(px)) - hw - search_px
        y0 = int(round(py)) - hh - search_px
        x1 = int(round(px)) + hw + search_px + 1
        y1 = int(round(py)) + hh + search_px + 1
        x0c, y0c, x1c, y1c = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        win = image[y0c:y1c, x0c:x1c]
        if win.shape[0] < th + 2 or win.shape[1] < tw + 2:
            continue
        res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, (j, i) = cv2.minMaxLoc(res)
        dx = dy = 0.0
        if 0 < j < res.shape[1] - 1:
            l, c, r = res[i, j - 1], res[i, j], res[i, j + 1]
            den = l - 2 * c + r
            if den < 0:
                dx = 0.5 * (l - r) / den
        if 0 < i < res.shape[0] - 1:
            u, c, d = res[i - 1, j], res[i, j], res[i + 1, j]
            den = u - 2 * c + d
            if den < 0:
                dy = 0.5 * (u - d) / den
        pos[k] = (x0c + j + hw + dx, y0c + i + hh + dy)
        scores[k] = float(score)
    return pos, scores


class LandmarkTracker(object):
    def __init__(self, name=None, params={}, color='green', bEqualizeHist=False):
        self.name = name
        self.color = color
        self.bgra = ui.bgra_dict[color]
        self.bgra_dim = tuple(0.5 * np.array(ui.bgra_dict[color]))
        self.bgra_state = ui.bgra_dict[color]
        self.bValidMask = True
        self.ptHinge_i = np.array([0.0, 0.0])
        self.angleBodypart_b = 0.0
        self.sense = 1
        self.iCount = 0
        self.dt = np.inf
        self.state = MsgState()
        self.state.angles = []
        self.state.radii = []
        self.state.gradients = []
        self.state.intensity = 0.0
        self.handles = {'hinge': ui.Handle(np.array([0.0, 0.0]), self.bgra, name='hinge')}
        self.params = {}
        self.ref_pts = np.zeros((0, 2))       # landmark positions at template capture
        self.templates = []                    # reference patches (template_px square)
        self.templates_big = []                # larger patches for rotated re-matching
        self.weak = []                         # landmark indices with textureless templates
        self.template_pts = None               # positions the templates were captured at
        self.last_fit = None                   # (theta, scale, dcentroid) for prediction
        self.matched = np.zeros((0, 2))
        self.scores = np.zeros(0)
        self.ok = np.zeros(0, dtype=bool)
        self.roll_s = None
        self.layout = {'n': 0, 'rms_radius': 0.0, 'min_dist': np.inf, 'ambiguity': np.inf}
        self.warnings = []                     # layout problems, shown in the GUI
        self.lever_px = None                   # set once yaw/pitch are calibrated
        self.resid_ema = None                  # per-landmark running residual
        self.set_params(params)

    # ------------------------------------------------------------ geometry

    def landmarks(self):
        return [(float(p['x']), float(p['y'])) for p in
                (self.params['gui'][self.name].get('landmarks') or [])]

    def set_params(self, params):
        self.params = params
        g = params['gui'][self.name]
        t = params[self.name]
        self.template_px = int(t.get('template_px', 17)) | 1
        self.search_px = int(t.get('search_px', 20))
        self.min_score = float(t.get('min_score', 0.5))
        self.similarity = t.get('fit', 'rigid') == 'similarity'
        self.rotate_templates = bool(t.get('rotate_templates', True))
        self.smooth = float(t.get('smooth', 0.0))
        self.nonrigid_px = float(t.get('nonrigid_px', 2.0))   # residual above this = warn
        lever = t.get('lever_px', None)                        # None until calibrated
        self.lever_px = float(lever) if lever else None
        self.view = str(params.get('view', 'frontal')).lower()
        if self.view not in VIEW_CHANNELS:
            self.view = 'frontal'
        self.channels = VIEW_CHANNELS[self.view]
        self.axis_deg = g.get('axis_deg', None)   # geometry: lives under gui.<part>
        self.ptHinge_i = np.array([g['hinge']['x'], g['hinge']['y']], dtype=float)
        pts = np.array(self.landmarks(), dtype=float).reshape(-1, 2)
        # Check the layout before trusting it, and never let the matcher search
        # far enough to reach a neighbouring landmark.
        self.layout = layout_metrics(pts)
        self.warnings = []
        safe = safe_search_px(self.layout['min_dist'], self.template_px)
        if safe is not None and self.search_px > safe:
            self.warnings.append('search %d->%d' % (self.search_px, safe))
            self.search_px = safe
        if len(pts) >= 2 and self.layout['ambiguity'] < 0.25 * self.layout['rms_radius']:
            self.warnings.append('AMBIGUOUS LAYOUT')
        if 0 < len(pts) < 4:
            self.warnings.append('only %d landmarks' % len(pts))
        # The part's long axis splits the displacement into pitch (along it)
        # and roll/yaw (across it). Prefer the explicit angle; fall back to
        # hinge->centroid, which is only meaningful with the hinge on the joint.
        self.axis, self.lateral = None, None
        if self.axis_deg is not None:
            self.axis, self.lateral = axis_vectors(self.axis_deg)
        elif len(pts):
            v = pts.mean(0) - self.ptHinge_i
            if np.linalg.norm(v) >= MIN_AXIS_PX:
                self.axis, self.lateral = axis_vectors(np.degrees(np.arctan2(v[1], v[0])))
            else:
                self.warnings.append('SET AXIS')
        # Keep templates only if the landmarks did not move.
        if self.template_pts is None or len(self.template_pts) != len(pts) or \
                (len(pts) and np.abs(self.template_pts - pts).max() > 0.5):
            self.templates, self.templates_big, self.template_pts = [], [], None
        self.ref_pts = pts
        # Half the landmark spread: only used to size the drawn axis line.
        self.draw_len = float(np.abs(pts - pts.mean(0)).max()) if len(pts) else 0.0
        self.last_fit = None
        self.roll_s = None
        self.update_handle_points()

    def axis_handle_len(self):
        return max(24.0, 1.4 * self.layout['rms_radius'])

    def update_handle_points(self):
        self.handles = {'hinge': ui.Handle(self.ptHinge_i.copy(), self.bgra, name='hinge')}
        if self.axis is not None:
            self.handles['axis'] = ui.Handle(
                self.ptHinge_i + self.axis * self.axis_handle_len(), self.bgra, name='axis')
        for k, (x, y) in enumerate(self.landmarks()):
            self.handles['lm%d' % k] = ui.Handle(np.array([x, y]), self.bgra, name='lm%d' % k)

    def hit_object(self, ptMouse):
        for tag, handle in self.handles.items():
            if tag != 'hinge' and handle.hit_test(ptMouse):
                return (self.name, tag)
        if self.handles['hinge'].hit_test(ptMouse):
            return (self.name, 'hinge')
        return (self.name, None)

    def create_mask(self, shape):
        pass

    def set_background(self, image):
        pass

    # ----------------------------------------------------------- templates

    def capture_templates(self, image):
        """Cut reference patches at the current landmark positions."""
        h = self.template_px // 2
        big = self.template_px + 2 * (h + 2)
        hb = big // 2
        H, W = image.shape[:2]
        self.templates, self.templates_big = [], []
        self.weak = []
        for k, (x, y) in enumerate(self.ref_pts):
            xi, yi = int(round(x)), int(round(y))
            if xi - hb < 0 or yi - hb < 0 or xi + hb + 1 > W or yi + hb + 1 > H:
                # Near the border: small template only if it fits, no rotation support.
                if xi - h < 0 or yi - h < 0 or xi + h + 1 > W or yi + h + 1 > H:
                    self.templates.append(None)
                    self.templates_big.append(None)
                    continue
                self.templates.append(image[yi - h:yi + h + 1, xi - h:xi + h + 1].copy())
                self.templates_big.append(None)
                continue
            self.templates_big.append(image[yi - hb:yi + hb + 1, xi - hb:xi + hb + 1].copy())
            self.templates.append(image[yi - h:yi + h + 1, xi - h:xi + h + 1].copy())
        # A template without texture (uniform patch) cannot be matched: flag it so
        # the user moves the landmark onto an edge or feature.
        for k, tpl in enumerate(self.templates):
            if tpl is not None and float(tpl.std()) < 4.0:
                self.weak.append(k)
                self.templates[k] = None
                self.templates_big[k] = None
        self.template_pts = self.ref_pts.copy()
        self.last_fit = None
        return sum(t is not None for t in self.templates)

    def current_positions(self):
        """Best estimate of where each landmark is right now: the matched
        position when the match was accepted, else the fit's prediction."""
        pos = self._predict()
        if len(self.matched) == len(pos) and len(self.ok) == len(pos):
            m = self.ok & np.isfinite(self.matched[:, 0])
            pos[m] = self.matched[m]
        return pos

    def zero(self, image=None):
        """Make the current posture the zero: recapture all templates."""
        if image is None:
            self.templates, self.templates_big, self.template_pts = [], [], None
            return 0
        return self.capture_templates(image)

    def _rotated_templates(self, theta):
        deg = -np.degrees(theta)          # cv2: positive = counter-clockwise on screen
        h = self.template_px // 2
        out = []
        for tpl, big in zip(self.templates, self.templates_big):
            if big is None:
                out.append(tpl)
                continue
            c = (big.shape[1] / 2.0 - 0.5, big.shape[0] / 2.0 - 0.5)
            M = cv2.getRotationMatrix2D(c, deg, 1.0)
            rot = cv2.warpAffine(big, M, (big.shape[1], big.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            cy, cx = big.shape[0] // 2, big.shape[1] // 2
            out.append(rot[cy - h:cy + h + 1, cx - h:cx + h + 1].copy())
        return out

    def save_templates(self, path):
        if self.template_pts is None:
            return False
        np.savez(path, pts=self.template_pts,
                 **{'t%d' % k: (t if t is not None else np.zeros((0, 0), np.uint8))
                    for k, t in enumerate(self.templates)},
                 **{'b%d' % k: (b if b is not None else np.zeros((0, 0), np.uint8))
                    for k, b in enumerate(self.templates_big)})
        return True

    def load_templates(self, path):
        """Load saved templates if they match the current landmarks."""
        if not os.path.exists(path):
            return False
        try:
            d = np.load(path)
            pts = d['pts']
        except Exception:
            return False
        if pts.shape != self.ref_pts.shape or (len(pts) and np.abs(pts - self.ref_pts).max() > 0.5):
            return False
        self.templates = [None if d['t%d' % k].size == 0 else d['t%d' % k] for k in range(len(pts))]
        self.templates_big = [None if d['b%d' % k].size == 0 else d['b%d' % k] for k in range(len(pts))]
        self.template_pts = pts.copy()
        return True

    # ------------------------------------------------------------ tracking

    def _predict(self):
        if self.last_fit is None:
            return self.ref_pts.copy()
        theta, scale, dc = self.last_fit
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        Pc = self.ref_pts.mean(0)
        return (scale * (R @ (self.ref_pts - Pc).T)).T + Pc + dc

    def _fit(self, image, templates):
        pos, scores = match_landmarks(image, templates, self._predict(), self.search_px,
                                      self.template_px // 2)
        ok = np.isfinite(pos[:, 0]) & (scores >= self.min_score)
        if ok.sum() < 2:
            return None, pos, scores, ok
        theta, scale, dc, resid = rigid_fit(self.ref_pts[ok], pos[ok], self.similarity)
        if ok.sum() >= 3:
            thr = max(3.0, 2.5 * float(np.median(resid)))
            keep = resid <= thr
            if keep.sum() >= 2 and keep.sum() < ok.sum():
                idx = np.where(ok)[0][~keep]
                ok[idx] = False
                theta, scale, dc, resid = rigid_fit(self.ref_pts[ok], pos[ok], self.similarity)
        # Spread the residuals back over all landmarks (NaN where unmatched) so
        # a single persistently bad point can be identified, not just the mean.
        full = np.full(len(self.ref_pts), np.nan)
        full[ok] = resid
        return (theta, scale, dc, float(resid.mean()), full), pos, scores, ok

    def update(self, dt, image, bInvertColor):
        self.iCount += 1
        self.dt = dt
        if not self.params['gui'][self.name]['track']:
            return
        if len(self.ref_pts) < 2:
            self.state.angles, self.state.radii, self.state.gradients = [], [], []
            self.state.intensity = 0.0
            return
        if self.template_pts is None:
            self.capture_templates(image)

        fit, pos, scores, ok = self._fit(image, self.templates)
        if fit is not None and self.rotate_templates and abs(fit[0]) > np.radians(4):
            fit2, pos2, scores2, ok2 = self._fit(image, self._rotated_templates(fit[0]))
            if fit2 is not None and ok2.sum() >= ok.sum() and fit2[3] <= fit[3] + 0.5:
                fit, pos, scores, ok = fit2, pos2, scores2, ok2
        self.matched, self.scores, self.ok = pos, scores, ok

        if fit is None:
            self.last_fit = None
            self.state.angles = [np.nan]
            self.state.radii = [np.nan, np.nan]
            self.state.gradients = [np.nan, np.nan, np.nan]
            self.state.kind = 'landmarks'
            self.state.extra = self._extra(np.nan, 0, None, np.nan)
            self.state.intensity = float(scores[np.isfinite(scores)].mean()) if np.isfinite(scores).any() else 0.0
            return
        theta, scale, dc, resid, resid_full = fit
        self.last_fit = (theta, scale, dc)
        roll = self.sense * theta
        # The head rolls about the neck (hinge), so a roll by itself displaces
        # the landmark centroid; remove that part, leaving the head's true
        # translation (mostly yaw from the front) in PIXELS. It is not turned
        # into an angle: the yaw lever arm is the head's anterior-posterior
        # depth, which a frontal view cannot measure.
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        arm = self.ref_pts.mean(0) - self.ptHinge_i
        dc_corr = dc - (R @ arm - arm)
        if self.smooth > 0:
            a = self.smooth
            self.roll_s = roll if self.roll_s is None else (1 - a) * roll + a * self.roll_s
            roll = self.roll_s
        self.state.angles = [float(roll)]
        self.state.radii = [float(dc[0]), float(dc[1])]
        self.state.gradients = [float(dc_corr[0]), float(dc_corr[1]), resid]
        self.state.intensity = float(scores[ok].mean())
        self.state.kind = 'landmarks'
        # Roll precision: a point localised to `resid` pixels, `rms_radius`
        # away from the centre of rotation, pins the angle to resid/radius;
        # averaging over the matched landmarks tightens it by sqrt(N).
        rms = self.layout['rms_radius']
        n_ok = int(ok.sum())
        sigma = resid / (rms * np.sqrt(n_ok)) if rms > 0 and n_ok else np.nan
        self._track_worst(resid_full)
        self.state.extra = self._extra(sigma, n_ok, dc_corr, roll)

    def _track_worst(self, resid_full):
        """Running per-landmark residual, so a single point that is quietly
        wrong (on a specular highlight, a leg, the proboscis) can be named.
        The mean residual hides it; this does not."""
        if resid_full is None or not len(resid_full):
            return
        if self.resid_ema is None or len(self.resid_ema) != len(resid_full):
            self.resid_ema = np.full(len(resid_full), np.nan)
        m = np.isfinite(resid_full)
        fresh = ~np.isfinite(self.resid_ema) & m
        self.resid_ema[fresh] = resid_full[fresh]
        upd = m & ~fresh
        self.resid_ema[upd] = 0.99 * self.resid_ema[upd] + 0.01 * resid_full[upd]

    def worst_landmark(self):
        """Index of the landmark that fits worst over time, or None."""
        if self.resid_ema is None or not np.isfinite(self.resid_ema).any():
            return None
        k = int(np.nanargmax(self.resid_ema))
        # Only call it out when it stands clearly apart from the others.
        others = np.delete(self.resid_ema, k)
        if not np.isfinite(others).any() or self.resid_ema[k] < 2.0 * np.nanmedian(others):
            return None
        return k

    def _extra(self, roll_sigma, n_ok, dc_corr, inplane):
        """The three rotations under fixed anatomical names, so downstream code
        never has to know the view.

        The in-plane one is always finite -- it is the fit's own rotation.
        The other two are the displacement resolved along the part's axis
        (pitch) and across it (roll or yaw), and stay NaN until `lever_px`
        turns pixels into an angle."""
        out = {'roll': np.nan, 'pitch': np.nan, 'yaw': np.nan,
               'd_axial': np.nan, 'd_lateral': np.nan,
               'roll_sigma': float(roll_sigma), 'n_ok': int(n_ok),
               'worst_lm': self.worst_landmark(), 'warnings': list(self.warnings),
               'view': self.view, 'channels': dict(self.channels)}
        out[self.channels['inplane']] = float(inplane)
        if dc_corr is None or self.axis is None:
            return out
        d_ax = float(np.dot(dc_corr, self.axis))
        d_lat = float(np.dot(dc_corr, self.lateral))
        out['d_axial'], out['d_lateral'] = d_ax, d_lat
        if self.lever_px:
            out[self.channels['axial']] = float(
                np.arcsin(np.clip(d_ax / self.lever_px, -1.0, 1.0)))
            out[self.channels['lateral']] = float(
                np.arcsin(np.clip(d_lat / self.lever_px, -1.0, 1.0)))
        return out

    # ------------------------------------------------------------- drawing

    def status_text(self):
        """The whole readout for this part, named for the camera view: the
        exact in-plane angle in degrees with its precision, then the two
        lever-arm channels (px, or degrees once calibrated), then match
        quality and any warnings. gui/app.py prefixes only the part label."""
        if not len(self.ok):
            return ''
        x = getattr(self.state, 'extra', None) or {}
        ch = self.channels
        inplane = x.get(ch['inplane'], np.nan)
        s = ' %-5s%+6.1f deg' % (ch['inplane'], np.degrees(inplane)) \
            if np.isfinite(inplane) else ' %-5s   -- ' % ch['inplane']
        sigma = x.get('roll_sigma', np.nan)
        if np.isfinite(sigma):
            s += ' +-%.1f' % np.degrees(sigma)
        d_ax, d_lat = x.get('d_axial', np.nan), x.get('d_lateral', np.nan)
        if np.isfinite(d_ax):
            if self.lever_px:
                s += '  %s%+5.1f %s%+5.1f deg' % (
                    ch['axial'], np.degrees(x.get(ch['axial'], np.nan)),
                    ch['lateral'], np.degrees(x.get(ch['lateral'], np.nan)))
            else:
                s += '  %s%+5.1f %s%+5.1f px' % (ch['axial'], d_ax,
                                                 ch['lateral'], d_lat)
        s += '  %d/%d  q%.2f' % (int(self.ok.sum()), len(self.ok), self.state.intensity)
        g = self.state.gradients
        if len(g) > 2 and np.isfinite(g[2]) and g[2] > self.nonrigid_px:
            s += '  NON-RIGID %.1fpx' % g[2]
        k = x.get('worst_lm')
        if k is not None:
            s += '  lm%d BAD' % k
        for w in self.warnings:
            s += '  ' + w
        return s

    def draw(self, image):
        g = self.params['gui'][self.name]
        self.handles['hinge'].draw(image)
        if not g['track']:
            return
        if self.axis is not None:
            # The long axis: displacement along it is pitch, across it is the
            # view's third rotation. Drawn so the convention is visible.
            tip = self.ptHinge_i + self.axis * self.axis_handle_len()
            cv2.arrowedLine(image, (int(self.ptHinge_i[0]), int(self.ptHinge_i[1])),
                            (int(tip[0]), int(tip[1])), (200, 200, 60), 1, tipLength=0.25)
            cv2.putText(image, '+' + self.channels['axial'], (int(tip[0]) + 3, int(tip[1]) - 3),
                        cv2.FONT_HERSHEY_DUPLEX, 0.32, (200, 200, 60))
        worst = (getattr(self.state, 'extra', None) or {}).get('worst_lm')
        for k, (x, y) in enumerate(self.ref_pts):
            if k in self.weak:
                cv2.circle(image, (int(x), int(y)), 6, (0, 0, 255), 2)   # no texture here
            elif k == worst:
                cv2.circle(image, (int(x), int(y)), 7, (0, 180, 255), 2)  # fits worst: move it
            else:
                cv2.circle(image, (int(x), int(y)), 4, self.bgra_dim, 1)  # reference position
        if len(self.matched):
            for k in range(len(self.matched)):
                x, y = self.matched[k]
                if not np.isfinite(x):
                    continue
                col = (0, 220, 0) if self.ok[k] else (0, 0, 255)
                cv2.drawMarker(image, (int(round(x)), int(round(y))), col,
                               cv2.MARKER_CROSS, 9, 1)
        if self.last_fit is not None and len(self.state.angles) >= 1 and np.isfinite(self.state.angles[0]):
            theta, scale, dc = self.last_fit
            c = self.ref_pts.mean(0) + dc
            L = max(15.0, 0.8 * self.draw_len)
            ax = np.array([np.cos(theta), np.sin(theta)]) * L
            cv2.line(image, (int(c[0] - ax[0]), int(c[1] - ax[1])),
                     (int(c[0] + ax[0]), int(c[1] + ax[1])), self.bgra, 1)
            cv2.arrowedLine(image, (int(self.ref_pts.mean(0)[0]), int(self.ref_pts.mean(0)[1])),
                            (int(c[0]), int(c[1])), (0, 200, 255), 1, tipLength=0.3)
            # The numeric readout (roll, landmarks matched, quality) is drawn with
            # the other parts' values in the bottom-left corner, by gui/app.py.
        elif len(self.ref_pts) < 2:
            cv2.putText(image, 'landmarks: place 2+ points (+lm)',
                        (int(self.ptHinge_i[0]) + 8, int(self.ptHinge_i[1]) - 8),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, self.bgra)
