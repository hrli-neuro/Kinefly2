"""Output names that never overwrite, and grouping recorded videos into jobs.

reserve_dir() is the overwrite protection for offline analysis. It CLAIMS a
folder by creating it, and creating a folder that already exists fails in the
operating system itself -- so a name in use is never handed out twice: not to a
second job in the same queue with the same name, and not to a second copy of the
program started at the same moment. A taken name gets _001, _002, ...

group_videos() pairs the files of one recording by the camera named in them:
run_rig writes <run>_bottom.avi and <run>_front.avi, and older recordings are
<run>_cam1 / <run>_cam2, where cam N is the N-th camera listed in rig.yaml.
"""
import os
import re

SEPARATORS = '_-. '
VIDEO_EXTENSIONS = ('.avi', '.mp4', '.mov', '.mkv', '.m4v', '.mpg', '.mpeg', '.wmv')


def safe_name(name, fallback='analysis'):
    """A name usable as a folder and file prefix on any OS: characters that
    are path separators or invalid somewhere become '_', and leading/trailing
    dots and spaces go (so nothing can climb out of the save folder)."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', '_', str(name or '')).strip(' .')
    return s or fallback


def candidates(base):
    """base, base_001, base_002, ..."""
    yield base
    n = 1
    while True:
        yield '%s_%03d' % (base, n)
        n += 1


def reserve_dir(parent, base):
    """Create and return a new, empty folder <parent>/<base> -- or
    <base>_001, <base>_002, ... when that name is taken. Never returns a
    folder that existed before the call."""
    os.makedirs(parent, exist_ok=True)
    for name in candidates(safe_name(base)):
        path = os.path.join(parent, name)
        try:
            os.mkdir(path)
        except FileExistsError:
            continue
        return path


def preview_name(parent, base, taken=()):
    """The name reserve_dir() would pick right now, without creating
    anything. `taken` holds names already promised to earlier jobs in the
    same queue. Compared case-insensitively, as the macOS and Windows
    filesystems do."""
    taken = {t.lower() for t in taken}
    for name in candidates(safe_name(base)):
        if name.lower() not in taken and not (parent and os.path.exists(os.path.join(parent, name))):
            return name


def camera_token(stem, cameras):
    """Which camera a file name says it came from, and the name without it.

    'run7_bottom' and 'run7_cam1' both give ('bottom', 'run7') when the rig
    lists bottom first. A token counts only between separators (_ - . space)
    or at the ends, so 'cam1' does not match 'cam10', and case is ignored. When
    several match, the one nearest the end wins -- that is where recording
    software puts it. Returns (None, stem) when no camera is named."""
    best = None
    for i, cam in enumerate(cameras):
        for token in (str(cam), 'cam%d' % (i + 1)):
            pat = r'(?:^|(?<=[%s]))%s(?=[%s]|$)' % (
                re.escape(SEPARATORS), re.escape(token), re.escape(SEPARATORS))
            for m in re.finditer(pat, stem, re.IGNORECASE):
                if best is None or m.start() > best[1].start():
                    best = (cam, m)
    if best is None:
        return None, stem
    cam, m = best
    head, tail = stem[:m.start()], stem[m.end():]
    kept = head.rstrip(SEPARATORS)
    joint = head[len(kept):][:1] if kept and tail.lstrip(SEPARATORS) else ''
    return cam, kept + joint + tail.lstrip(SEPARATORS)


def group_videos(paths, cameras, primary=None):
    """Turn a list of video files into analysis jobs.

    Files in the same folder whose names differ only in the camera token
    become ONE job with a video per camera -- the bottom+front pair of one
    recording. Every other file is a job of its own, on the camera its name
    says, or else on the primary camera. A second file for a camera a group
    already has starts a new job rather than replacing the first.

    Returns [{'name': ..., 'videos': {camera: absolute path}}] in the order
    the files were given; the name is the file name without the camera."""
    cameras = [str(c) for c in cameras]
    primary = primary if primary in cameras else (cameras[0] if cameras else 'camera')
    jobs, groups = [], {}
    for p in paths:
        p = os.path.abspath(str(p))
        stem = os.path.splitext(os.path.basename(p))[0]
        cam, rest = camera_token(stem, cameras)
        if cam is None:
            jobs.append({'name': safe_name(stem), 'videos': {primary: p}})
            continue
        key = (os.path.dirname(p), rest.lower())
        job = groups.get(key)
        if job is None or cam in job['videos']:
            job = {'name': safe_name(rest or stem), 'videos': {}}
            groups[key] = job
            jobs.append(job)
        job['videos'][cam] = p
    # Present each job's cameras in rig order (bottom before front).
    for job in jobs:
        job['videos'] = {c: job['videos'][c] for c in cameras if c in job['videos']} \
            or job['videos']
    return jobs
