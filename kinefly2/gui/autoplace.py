"""Automatic placement of the tracking regions from the camera image.

Tier A — registration to a reference (primary). When the user saves a tuned
config, the frame it was tuned on is stored as reference_<camera>.png. For a
new fly, estimate the rigid transform (rotation + translation) between that
reference and the current frame and move every hinge/center/axis point with
it. Body-frame angle windows (angle_lo/hi) are relative to the body axis and
therefore unchanged; only the aux ellipse's image-frame angle rotates.

Tier B — silhouette bootstrap (no reference yet; bottom view). Threshold the
fly (polarity from invertcolor), take the largest blob, and place the body
axis and hinges from its image moments. Deliberately rough: the tether merges
with the body in the silhouette. The user refines once and saves; Tier A then
takes over.

All functions are pure (image + params dict in, params dict out).
"""
import copy
import math

import cv2
import numpy as np
from ..tracking import parts

POINT_KEYS = (('gui', 'head', 'hinge'), ('gui', 'abdomen', 'hinge'),
              ('gui', 'left', 'hinge'), ('gui', 'right', 'hinge'),
              ('gui', 'aux', 'center'), ('gui', 'axis', 'pt1'), ('gui', 'axis', 'pt2'))


def _get(d, keys):
    for k in keys:
        d = d[k]
    return d


def _all_points(params):
    """Every image-space point dict in the params: hinges/centers/axis points
    and any per-part landmark lists."""
    pts = [_get(params, keys) for keys in POINT_KEYS]
    for part in parts.body_parts(params):
        pts.extend(params['gui'][part].get('landmarks') or [])
    return pts


def body_mask(params, shape, margin=1.0):
    """Disk around the body (between head and abdomen hinges), used to focus
    the registration on the fly rather than on the wings/background."""
    h = params['gui']['head']['hinge']
    a = params['gui']['abdomen']['hinge']
    cx, cy = (h['x'] + a['x']) / 2.0, (h['y'] + a['y']) / 2.0
    L = math.hypot(h['x'] - a['x'], h['y'] - a['y'])
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(max(20, margin * L)), 255, -1)
    return mask


def register(ref, cur, mask=None, allow_scale=False):
    """Rigid transform W (2x3) mapping reference pixel coordinates to current
    ones: p_cur = W @ [p_ref, 1]. Returns (W, method) with method in
    {'ecc', 'phase'}; falls back to translation-only phase correlation if ECC
    does not converge."""
    ref32 = cv2.GaussianBlur(ref, (0, 0), 1.5).astype(np.float32)
    cur32 = cv2.GaussianBlur(cur, (0, 0), 1.5).astype(np.float32)
    motion = cv2.MOTION_AFFINE if allow_scale else cv2.MOTION_EUCLIDEAN
    W = np.eye(2, 3, dtype=np.float32)
    # Seed the rotation-capable search with the translation from phase correlation.
    (dx, dy), _ = cv2.phaseCorrelate(ref32, cur32)
    W[0, 2], W[1, 2] = dx, dy
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
    try:
        # ECC: input warped by W matches template  ->  template=ref, input=cur
        # would give cur(W x) = ref(x), i.e. W maps ref coords to cur coords.
        _, W = cv2.findTransformECC(ref32, cur32, W, motion, criteria,
                                    mask if mask is not None else None, 5)
        return W, 'ecc'
    except cv2.error:
        W = np.eye(2, 3, dtype=np.float32)
        W[0, 2], W[1, 2] = dx, dy
        return W, 'phase'


def transform_params(params, W):
    """Apply W to every image-space point in the params (unscaled pixels)."""
    out = copy.deepcopy(params)
    for pt in _all_points(out):
        v = W @ np.array([pt['x'], pt['y'], 1.0])
        pt['x'], pt['y'] = float(v[0]), float(v[1])
    rot = math.atan2(W[1, 0], W[0, 0])
    out['gui']['aux']['angle'] = float(out['gui']['aux']['angle'] + rot)
    return out, rot


def silhouette_bootstrap(img, params, invertcolor=True):
    """Place body axis and hinges from the fly's silhouette (bottom view)."""
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    thr = cv2.THRESH_BINARY_INV if invertcolor else cv2.THRESH_BINARY
    _, bw = cv2.threshold(blur, 0, 255, thr | cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
    if n < 2:
        raise ValueError('No fly silhouette found')
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = (labels == i).astype(np.uint8)
    m = cv2.moments(blob, binaryImage=True)
    cx, cy = m['m10'] / m['m00'], m['m01'] / m['m00']
    mu20, mu02, mu11 = m['mu20'] / m['m00'], m['mu02'] / m['m00'], m['mu11'] / m['m00']
    theta = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
    common = math.sqrt(max(0.0, (mu20 - mu02) ** 2 + 4 * mu11 ** 2))
    l1 = math.sqrt(max(1e-6, (mu20 + mu02 + common) / 2))
    l2 = math.sqrt(max(1e-6, (mu20 + mu02 - common) / 2))
    length, width = 4.0 * l1, 4.0 * l2
    ux, uy = math.cos(theta), math.sin(theta)

    # Which end is the head? The tether enters from an image border; the head
    # is the end pointing away from the border the blob touches.
    H, W_ = blob.shape
    x0, y0, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                    stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
    touches = {'left': x0 <= 0, 'right': x0 + w >= W_, 'top': y0 <= 0, 'bottom': y0 + h >= H}
    away = np.array([0.0, 0.0])
    if touches['left']:
        away += [1, 0]
    if touches['right']:
        away += [-1, 0]
    if touches['top']:
        away += [0, 1]
    if touches['bottom']:
        away += [0, -1]
    if np.dot(away, [ux, uy]) < 0:
        ux, uy = -ux, -uy

    out = copy.deepcopy(params)
    g = out['gui']
    px, py = -uy, ux            # perpendicular (image y is down)
    g['head']['hinge'] = {'x': float(cx + 0.40 * length * ux), 'y': float(cy + 0.40 * length * uy)}
    g['abdomen']['hinge'] = {'x': float(cx - 0.40 * length * ux), 'y': float(cy - 0.40 * length * uy)}
    # 'left' is the fly's left: with the head along +u, left is at -perp
    # in image coordinates when y points down (perp = rotate u by +90 deg).
    g['left']['hinge'] = {'x': float(cx - 0.35 * width * px), 'y': float(cy - 0.35 * width * py)}
    g['right']['hinge'] = {'x': float(cx + 0.35 * width * px), 'y': float(cy + 0.35 * width * py)}
    for part in ('left', 'right'):
        g[part]['radius_inner'] = float(0.6 * width)
        g[part]['radius_outer'] = float(1.2 * length)
    g['head']['radius_inner'] = float(0.15 * width)
    g['head']['radius_outer'] = float(0.6 * width)
    g['abdomen']['radius_inner'] = float(0.15 * width)
    g['abdomen']['radius_outer'] = float(0.5 * width)
    g['axis']['pt1'] = dict(g['head']['hinge'])
    g['axis']['pt2'] = dict(g['abdomen']['hinge'])
    return out, {'centroid': (cx, cy), 'angle': math.atan2(uy, ux), 'length': length, 'width': width}


# --- safety -----------------------------------------------------------------

def plausible_transform(W, shape, max_shift_frac=0.25, max_rot_deg=30.0):
    """Reject registrations that cannot be a re-mounted fly: large shifts
    (relative to the image size) or large rotations."""
    h, w = shape[:2]
    rot = abs(math.degrees(math.atan2(W[1, 0], W[0, 0])))
    shift = math.hypot(W[0, 2], W[1, 2])
    return shift <= max_shift_frac * max(h, w) and rot <= max_rot_deg


def points_inside(params, shape, margin=2):
    """True if every hinge/center/axis point lies inside the image."""
    h, w = shape[:2]
    for pt in _all_points(params):
        if not (margin <= pt['x'] <= w - 1 - margin and margin <= pt['y'] <= h - 1 - margin):
            return False
    return True


def clamp_to_image(params, shape, margin=2):
    """Clamp every point into the image so handles always stay reachable."""
    out = copy.deepcopy(params)
    h, w = shape[:2]
    for pt in _all_points(out):
        pt['x'] = float(min(max(pt['x'], margin), w - 1 - margin))
        pt['y'] = float(min(max(pt['y'], margin), h - 1 - margin))
    return out
