"""RotationTracker: rotation of a bodypart about its hinge by template
rotation search (coarse-to-fine, sub-degree), for small rotations that polar
phase correlation cannot resolve -- e.g. head roll seen from the front, where
the head is a mostly-symmetric bright ball with faint internal texture.

Built on AreaTracker: same ROI/mask/background handling, state convention
(state.angles[0] in the bodypart frame, radians), autozero re-anchoring, and
drawing. Only the measurement differs: the reference ROI is rotated about the
hinge by candidate angles and compared with the current ROI by normalized
correlation over the mask; the best angle (parabolic-refined) is the rotation.

Per-part parameters (tracker: section): search_deg (default 8) -- half-range
of the search; cost grows with it (rotated references are cached, so the
per-frame cost is ~search_deg/step correlations plus ~8 fine warps).
"""
import cv2
import numpy as np

from .tracker_area import AreaTracker


class RotationTracker(AreaTracker):
    def __init__(self, name=None, params={}, color='white', bEqualizeHist=False):
        AreaTracker.__init__(self, name, params, color, bEqualizeHist)
        self.imgRef = None
        self._bank = {}
        self._bank_ref_id = None
        self._idx = None
        self._idx_shape = None

    def set_params(self, params):
        AreaTracker.set_params(self, params)
        self.imgRef = None
        self._bank = {}
        self.search_deg = float(self.params[self.name].get('search_deg', 8.0))
        self.coarse_step = 1.0 if self.search_deg <= 10 else 2.0

    def zero(self):
        AreaTracker.zero(self)
        self.imgRef = None
        self._bank = {}

    # --- measurement -------------------------------------------------------

    def _rotate(self, img, deg):
        M = cv2.getRotationMatrix2D((float(self.j_0), float(self.i_0)), deg, 1.0)
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def _ncc(self, a, b):
        a = a[self._idx]
        b = b[self._idx]
        a = a - a.mean()
        b = b - b.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 0 else -1.0

    def _measure(self, cur):
        """Rotation (deg, cv2 convention: positive = counter-clockwise on
        screen) that best maps the reference onto the current ROI."""
        if self._bank_ref_id != id(self.imgRef):
            self._bank = {}
            self._bank_ref_id = id(self.imgRef)
        coarse = np.arange(-self.search_deg, self.search_deg + 1e-6, self.coarse_step)
        for deg in coarse:
            if deg not in self._bank:
                self._bank[deg] = self._rotate(self.imgRef, float(deg))
        scores = np.array([self._ncc(self._bank[d], cur) for d in coarse])
        best = float(coarse[int(np.argmax(scores))])

        fine = best + np.arange(-self.coarse_step, self.coarse_step + 1e-6, self.coarse_step / 5.0)
        fscores = np.array([self._ncc(self._rotate(self.imgRef, float(d)), cur) for d in fine])
        k = int(np.argmax(fscores))
        best = float(fine[k])
        if 0 < k < len(fine) - 1:                       # Parabolic sub-step refinement.
            y0, y1, y2 = fscores[k - 1], fscores[k], fscores[k + 1]
            denom = (y0 - 2 * y1 + y2)
            if denom < 0:
                best += 0.5 * (y0 - y2) / denom * (fine[1] - fine[0])
        return best, float(fscores[k])

    def update_state(self):
        # Compare UNMASKED ROIs (the mask only selects which pixels count):
        # a mask baked into the images would be a static high-contrast edge
        # that pins the best rotation at zero on low-texture bodyparts.
        img = self.imgRoiFg
        if img is None or self.mask.img is None or img.shape[:2] != self.mask.img.shape[:2]:
            return
        cur = img.astype(np.float32)
        if self._idx is None or self._idx_shape != cur.shape:
            self._idx = self.mask.img > 0
            self._idx_shape = cur.shape
            self.imgRef = None
        if self.imgRef is None or self.imgRef.shape != cur.shape:
            self.imgRef = cur.copy()
            self.state.angles = [self.stateOrigin_p.angles[0]]
            self.state.radii = [self.stateOrigin_p.radii[0]]
            return

        deg, score = self._measure(cur)
        # cv2 positive = counter-clockwise on screen = decreasing image angle
        # (y down); the area tracker's convention is positive with image angle.
        angleOffset = self.sense * (-np.radians(deg))
        self.state.angles = [((self.stateOrigin_p.angles[0] + angleOffset + np.pi) % (2 * np.pi)) - np.pi]
        self.state.radii = [self.stateOrigin_p.radii[0]]
        self.state.gradients = [score]

        self.stateLo_p.angles = [min(self.stateLo_p.angles[0], self.state.angles[0])]
        self.stateHi_p.angles = [max(self.stateHi_p.angles[0], self.state.angles[0])]

        # Autozero (same policy as AreaTracker): when the current angle lies
        # between the origin and the midpoint of the observed range, take the
        # current ROI as the new reference and converge the origin to it.
        if self.params[self.name]['autozero'] and self.iCount > 100:
            angleRef_p = (self.stateHi_p.angles[0] + self.stateLo_p.angles[0]) / 2.0
            a, o = self.state.angles[0], self.stateOrigin_p.angles[0]
            if (angleRef_p < a < o) or (o < a < angleRef_p):
                self.imgRef = cur.copy()
                self.stateLo_p.angles[0] -= angleOffset
                self.stateHi_p.angles[0] -= angleOffset
                self.state.angles[0] = o

        if self.imgRoiFgMasked is not None:
            self.state.intensity = float(np.sum(self.imgRoiFgMasked) / self.mask.sum)
