"""LandmarkTracker: head roll AND yaw from user-placed landmarks.

The user places a few landmarks on distinctive head features (eye edges, the
bright inter-ocular patch, antenna bases). Each frame, every landmark is
located by normalized template matching of its reference patch inside a
small search window; a rigid (or similarity) fit of the reference landmark
positions onto the measured ones decomposes the head motion:

    rotation of the fit          -> roll   (in-plane rotation)
    lateral translation of the fit -> yaw  (from the front, yaw shifts the
                                            head sideways), after removing the
                                            centroid shift that a roll about the
                                            neck hinge produces; converted to an
                                            angle with the neck->head radius

Unlike the hinge-based trackers (edge/area/rotation) this needs no rotation
center and is not corrupted by combined roll+yaw. Blob-like features also
survive soft focus. Cost: N small matchTemplate calls, well under 1 ms.

Outputs (bodypart frame, like the other trackers):
    state.angles    = [roll, yaw]           radians
    state.radii     = [dx, dy]              head centroid shift, px
    state.gradients = [residual_px, axial]  fit residual; along-axis shift (pitch/approach)
    state.intensity = mean match score      0..1 quality
Per-part parameters (tracker: section): template_px, search_px, min_score,
fit ('rigid'|'similarity'), rotate_templates, smooth, nonrigid_px (residual
above which the overlay warns that the landmarks do not move as one body:
landmarks on specular highlights, legs, or the proboscis, or a mismatch).
Geometry (gui.<part>): hinge (neck), landmarks: [{x, y}, ...], yaw_radius.
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
        self.yaw_s = None
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
        self.ptHinge_i = np.array([g['hinge']['x'], g['hinge']['y']], dtype=float)
        # Body axis (abdomen -> head) defines lateral (yaw) vs axial directions.
        h = params['gui']['head']['hinge']
        a = params['gui']['abdomen']['hinge']
        ax = np.array([h['x'] - a['x'], h['y'] - a['y']], dtype=float)
        n = np.linalg.norm(ax)
        self.axial = ax / n if n > 0 else np.array([0.0, -1.0])
        self.lateral = np.array([-self.axial[1], self.axial[0]])
        pts = np.array(self.landmarks(), dtype=float).reshape(-1, 2)
        # Keep templates only if the landmarks did not move.
        if self.template_pts is None or len(self.template_pts) != len(pts) or \
                (len(pts) and np.abs(self.template_pts - pts).max() > 0.5):
            self.templates, self.templates_big, self.template_pts = [], [], None
        self.ref_pts = pts
        self.yaw_radius = float(g.get('yaw_radius') or 0.0)
        if self.yaw_radius <= 0 and len(pts):
            self.yaw_radius = float(np.linalg.norm(pts.mean(0) - self.ptHinge_i)) or 1.0
        self.last_fit = None
        self.roll_s = self.yaw_s = None
        self.update_handle_points()

    def update_handle_points(self):
        self.handles = {'hinge': ui.Handle(self.ptHinge_i.copy(), self.bgra, name='hinge')}
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
        return (theta, scale, dc, float(resid.mean())), pos, scores, ok

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
            self.state.angles = [np.nan, np.nan]
            self.state.radii = [np.nan, np.nan]
            self.state.gradients = [np.nan, np.nan]
            self.state.intensity = float(scores[np.isfinite(scores)].mean()) if np.isfinite(scores).any() else 0.0
            return
        theta, scale, dc, resid = fit
        self.last_fit = (theta, scale, dc)
        roll = self.sense * theta
        # The head rolls about the neck (hinge), so a roll by itself displaces
        # the landmark centroid; remove that part before reading yaw from the
        # remaining lateral shift.
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        arm = self.ref_pts.mean(0) - self.ptHinge_i
        dc_yaw = dc - (R @ arm - arm)
        lateral = float(dc_yaw @ self.lateral)
        axial = float(dc_yaw @ self.axial)
        yaw = float(np.arctan2(lateral, self.yaw_radius))
        if self.smooth > 0:
            a = self.smooth
            self.roll_s = roll if self.roll_s is None else (1 - a) * roll + a * self.roll_s
            self.yaw_s = yaw if self.yaw_s is None else (1 - a) * yaw + a * self.yaw_s
            roll, yaw = self.roll_s, self.yaw_s
        self.state.angles = [float(roll), yaw]
        self.state.radii = [float(dc[0]), float(dc[1])]
        self.state.gradients = [resid, axial]
        self.state.intensity = float(scores[ok].mean())

    # ------------------------------------------------------------- drawing

    def draw(self, image):
        g = self.params['gui'][self.name]
        self.handles['hinge'].draw(image)
        if not g['track']:
            return
        for k, (x, y) in enumerate(self.ref_pts):
            if k in self.weak:
                cv2.circle(image, (int(x), int(y)), 6, (0, 0, 255), 2)   # no texture here
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
        if self.last_fit is not None and len(self.state.angles) == 2 and np.isfinite(self.state.angles[0]):
            theta, scale, dc = self.last_fit
            c = self.ref_pts.mean(0) + dc
            L = max(15.0, 0.6 * self.yaw_radius)
            ax = np.array([np.cos(theta), np.sin(theta)]) * L
            cv2.line(image, (int(c[0] - ax[0]), int(c[1] - ax[1])),
                     (int(c[0] + ax[0]), int(c[1] + ax[1])), self.bgra, 1)
            cv2.arrowedLine(image, (int(self.ref_pts.mean(0)[0]), int(self.ref_pts.mean(0)[1])),
                            (int(c[0]), int(c[1])), (0, 200, 255), 1, tipLength=0.3)
            txt = 'roll %+.1f  yaw %+.1f  %d/%d  q%.2f' % (
                np.degrees(self.state.angles[0]), np.degrees(self.state.angles[1]),
                int(self.ok.sum()), len(self.ok), self.state.intensity)
            if self.state.gradients and self.state.gradients[0] > self.nonrigid_px:
                txt += '  NON-RIGID %.1fpx' % self.state.gradients[0]   # landmarks not moving together
            cv2.putText(image, txt, (int(c[0]) + 8, int(c[1]) - 8), cv2.FONT_HERSHEY_DUPLEX,
                        0.4, self.bgra)
        elif len(self.ref_pts) < 2:
            cv2.putText(image, 'landmarks: place 2+ points (+lm)',
                        (int(self.ptHinge_i[0]) + 8, int(self.ptHinge_i[1]) - 8),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, self.bgra)
