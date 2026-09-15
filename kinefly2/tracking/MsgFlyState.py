#!/usr/bin/env python

class MsgFlyState:
    def __init__(self):
        self.head           = []
        self.abdomen        = []
        self.left           = []
        self.right          = []
        self.aux            = []
        # Which parts this message actually carries. A camera may declare extra
        # parts (hind legs on the ventral view), so consumers read this rather
        # than assuming the standard five.
        self.partnames      = ('head', 'abdomen', 'left', 'right', 'aux')

class MsgState:
    def __init__(self):
        self.angles         = []
        self.gradients      = []
        self.radii          = []
        self.freq           = []
        self.intensity      = []
        self.kind           = None   # Tracker type, when consumers need to know.
        self.extra          = {}     # Tracker-specific named channels.

class Header:
    def __init__(self, seq=0, stamp=0.0, frame_id=''):
        self.seq            = seq
        self.stamp          = stamp      # Seconds (float); camera timestamp when available.
        self.frame_id       = frame_id

class Rosimg:
    def __init__(self, header, height, width, encoding, is_bigendian, step, data):
        self.header         = Header
        self.height         = []
        self.width          = []
        self.encoding       = []
        self.is_bigendian   = []
        self.step           = []
        self.data           = []