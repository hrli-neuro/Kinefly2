"""What produced a recording: code version and measurement conventions.

A recording is only re-analysable if you know which code wrote it. The camera
YAMLs are snapshotted into every recording folder, but a YAML does not pin the
CONVENTION behind the numbers: on 2026-09-10 the wing zero changed from the
perpendicular of the head-abdomen axis to the line joining the two wing bases,
which shifts every wing angle by ~15 degrees while the config is byte-identical.
Two recordings with the same YAML can therefore mean different things.

So every recording carries provenance(): the git description of the working
tree, the package version, and the conventions in force. Nothing here can fail
a recording -- a copy with no git history still records, it just says so.
"""
import os
import platform
import socket
import subprocess

from . import __version__

# Conventions the stored numbers depend on. Bump a value here whenever the
# meaning of a logged channel changes, even if nothing else does; analysis can
# then branch on the recording instead of on the date.
CONVENTIONS = {
    # 0 deg for each wing is the line joining the two wing hinges, shared by
    # both wings, so a symmetric posture reads L = R (README 2.5.5). Before
    # 2026-09-10 this was 'body_axis_perpendicular' and the head/abdomen hinge
    # positions leaked into the wing angles.
    'wing_zero': 'wing_base_line',
    # The head's zero is a captured posture: the landmark templates, recaptured
    # by the `zero` button.
    'head_zero': 'landmark_templates',
    # Everything stored or streamed is radians; only the GUI/dashboard convert.
    'angle_units': 'radians',
}

_cached = None


def code_version():
    """`git describe` of the checkout this package lives in, e.g.
    'fb49a07' or 'fb49a07-dirty'; falls back to the package version when there
    is no git history (an installed copy, a tarball). Never raises."""
    global _cached
    if _cached is not None:
        return _cached
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(['git', '-C', here, 'describe', '--tags', '--always', '--dirty'],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             timeout=2, check=True)
        _cached = out.stdout.decode().strip() or __version__
    except Exception:
        # No git, not a repo, git too slow: the version is still worth having.
        _cached = 'v%s (no git)' % __version__
    return _cached


def is_dirty():
    """True when the recording code has uncommitted changes, so its
    `code_version` cannot be checked out again exactly."""
    return code_version().endswith('-dirty')


def provenance():
    """The block stamped into every recording (HDF5 attrs, _experiment.yaml)."""
    return {'code_version': code_version(),
            'kinefly2_version': __version__,
            'conventions': dict(CONVENTIONS),
            'python': platform.python_version(),
            'host': socket.gethostname()}
