"""Serializable fly-state containers, replacing Kinefly's ROS MsgFlystate.

A CameraState holds one camera's per-frame measurements; the StateMerger
combines the latest CameraState from each camera into a MergedFlyState, which
is what the output sinks (ZMQ, logger, and eventually the LED-panel driver)
consume.
"""
import math
from dataclasses import dataclass, field


def _first(seq, default=math.nan):
    try:
        return float(seq[0])
    except (IndexError, TypeError):
        return default


@dataclass
class BodypartState:
    angles: list = field(default_factory=list)     # Radians; [0] is the primary edge/angle.
    gradients: list = field(default_factory=list)
    radii: list = field(default_factory=list)      # Pixels.
    freq: float = 0.0                              # Wingbeat frequency (aux tracker), Hz.
    intensity: float = 0.0

    @classmethod
    def from_msg(cls, msg):
        def listify(x):
            if x is None:
                return []
            if isinstance(x, (int, float)):
                return [float(x)]
            return [float(v) for v in x]
        freq = msg.freq if isinstance(msg.freq, (int, float)) else 0.0
        intensity = msg.intensity if isinstance(msg.intensity, (int, float)) else 0.0
        return cls(angles=listify(msg.angles), gradients=listify(msg.gradients),
                   radii=listify(msg.radii), freq=float(freq), intensity=float(intensity))

    @property
    def angle(self):
        return _first(self.angles)

    @property
    def radius(self):
        return _first(self.radii)

    def to_dict(self):
        return {'angles': self.angles, 'gradients': self.gradients,
                'radii': self.radii, 'freq': self.freq, 'intensity': self.intensity}


@dataclass
class CameraState:
    camera: str          # e.g. 'bottom', 'front'.
    seq: int
    t_camera: float      # Device timestamp (s, device clock; NaN if unavailable).
    t_host: float        # Host monotonic time at frame receipt.
    t_done: float        # Host monotonic time when tracking finished.
    parts: dict = field(default_factory=dict)   # name -> BodypartState

    @classmethod
    def from_flystate_msg(cls, camera, msg, frame, t_done):
        parts = {name: BodypartState.from_msg(getattr(msg, name))
                 for name in ('head', 'abdomen', 'left', 'right', 'aux')}
        return cls(camera=camera, seq=msg.header.seq, t_camera=frame.t_camera,
                   t_host=frame.t_host, t_done=t_done, parts=parts)

    def to_dict(self):
        return {'camera': self.camera, 'seq': self.seq, 't_camera': self.t_camera,
                't_host': self.t_host, 't_done': self.t_done,
                'parts': {k: v.to_dict() for k, v in self.parts.items()}}


@dataclass
class MergedFlyState:
    seq: int
    t_host: float                                # Host monotonic time of the merge.
    cameras: dict = field(default_factory=dict)  # camera name -> CameraState

    def part(self, camera, name):
        cs = self.cameras.get(camera)
        if cs is None:
            return None
        return cs.parts.get(name)

    def to_dict(self):
        return {'seq': self.seq, 't_host': self.t_host,
                'cameras': {k: v.to_dict() for k, v in self.cameras.items()}}
