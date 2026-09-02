"""Per-camera tracker view: owns the params, the Fly, and the Kinefly-style
OpenCV GUI (toolbar buttons + draggable hinge/angle/radius handles).

Ported from Benifly's MainWindow (itself from Kinefly's kinefly.py), with the
ROS/file plumbing replaced: frames come in through process(), parameters live
in a per-camera YAML file, and results go out as CameraState objects.

Threading contract: process() and render() run in the camera's pipeline thread;
on_mouse() runs in the GUI (main) thread. All three take self.lock. While the
user is dragging (bMousing), process() skips fly.update, same as Kinefly.
"""
import copy
import os
import threading
import time

import cv2
import numpy as np
import yaml

from ..core.flystate import CameraState
from . import autoplace
from ..tracking import ui
from ..tracking import imageprocessing
from ..tracking.fly import Fly
from ..tracking.setdict import SetDict
from ..tracking.MsgFlyState import Header


def tracker_defaults():
    return {'invertcolor': 'auto',   # 'auto', true (dark fly on bright bg), or false.
            'scale_image': 1.0,      # Reducing the image scale will speed the framerate.
            'n_edges_max': 1,        # Max number of edges per wing to detect, subject to threshold.
            'rc_background': 1000.0, # Time constant (s) of the moving average background.
            'wingbeat_min': 180,     # Bounds for wingbeat frequency measurement.
            'wingbeat_max': 220,
            'head': {'tracker': 'area',
                     'autozero': True,
                     'threshold': 0.0,
                     'feathering': 0.25,
                     'saturation_correction': True},
            'abdomen': {'tracker': 'tip',
                        'autozero': True,
                        'threshold': 0.0,
                        'feathering': 0.25,
                        'saturation_correction': True},
            'left': {'tracker': 'edge',
                     'autozero': True,
                     'threshold': 0.0,
                     'feathering': 0.25,
                     'saturation_correction': True},
            'right': {'tracker': 'edge',
                      'autozero': True,
                      'threshold': 0.0,
                      'feathering': 0.25,
                      'saturation_correction': True},
            'aux': {'tracker': 'intensity',
                    'saturation_correction': True},
            'gui': {'windows': False,
                    'symmetric': True,
                    'axis': {'track': False,
                             'pt1': {'x': 175, 'y': 50},
                             'pt2': {'x': 175, 'y': 150}},
                    'head': {'track': True,
                             'subtract_bg': False,
                             'stabilize': False,
                             'hinge': {'x': 175, 'y': 90},
                             'radius_outer': 20,
                             'radius_inner': 10,
                             'angle_hi': 0.7854,
                             'angle_lo': -0.7854},
                    'abdomen': {'track': False,
                                'subtract_bg': False,
                                'stabilize': False,
                                'hinge': {'x': 175, 'y': 150},
                                'radius_outer': 10,
                                'radius_inner': 5,
                                'angle_hi': 3.927,
                                'angle_lo': 2.3562},
                    'left': {'track': True,
                             'subtract_bg': False,
                             'stabilize': False,
                             'hinge': {'x': 150, 'y': 150},
                             'radius_outer': 20,
                             'radius_inner': 10,
                             'angle_hi': -0.7854,
                             'angle_lo': -2.3562},
                    'right': {'track': True,
                              'subtract_bg': False,
                              'stabilize': False,
                              'hinge': {'x': 200, 'y': 150},
                              'radius_outer': 20,
                              'radius_inner': 10,
                              'angle_hi': 2.3562,
                              'angle_lo': 0.7854},
                    'aux': {'track': False,
                            'subtract_bg': False,
                            'center': {'x': 50, 'y': 50},
                            'radius1': 30,
                            'radius2': 20,
                            'angle': 0.0},
                    }
            }


class TrackerView:
    def __init__(self, name, config_path, use_gui=True):
        self.name = name
        self.config_path = str(config_path)
        self.use_gui = use_gui
        self.window_name = 'kinefly2: %s' % name
        self.lock = threading.RLock()
        self.bQuit = False

        with open(self.config_path) as f:
            self.config = yaml.safe_load(f) or {}

        self.params = self.config.get('tracker') or {}
        SetDict().set_dict_with_preserve(self.params, tracker_defaults())
        self.params = self.legalize_params(self.params)
        self.scale = self.params['scale_image']
        self.paramsStartup = copy.deepcopy(self.params)   # 'reset' target
        self.paramsUndo = None                             # params before the last autoplace

        self.bMousing = False
        self.bValidImage = False
        self.fly = Fly(self.scale_params(self.params, self.scale))
        self.fly.set_params(self.scale_params(self.params, self.scale))

        # invertcolor: 'auto' detects from the first frame (dark fly on bright
        # backlight vs bright fly on dark background); an explicit true/false
        # from the YAML (or the GUI toggle, which persists) overrides it.
        inv = self.params.get('invertcolor', 'auto')
        if inv in ('auto', None):
            self.fly.bInvertColorAuto = True
            self.fly.bInvertColorValid = False
        else:
            self.fly.bInvertColor = bool(inv)
            self.fly.bInvertColorValid = True
            self.fly.bInvertColorAuto = False

        # Background image for subtraction, stored next to the config file.
        cfg_dir = os.path.dirname(os.path.abspath(self.config_path))
        self.filenameBackground = os.path.join(cfg_dir, 'background_%s.png' % name)
        self.filenameReference = os.path.join(cfg_dir, 'reference_%s.png' % name)
        self.statusText = ''
        imgDisk = cv2.imread(self.filenameBackground, cv2.IMREAD_GRAYSCALE)
        if imgDisk is not None:
            if self.scale != 1.0:
                imgDisk = cv2.resize(imgDisk, (0, 0), fx=self.scale, fy=self.scale)
            self.fly.set_background(imgDisk)
            self.bHaveBackground = True
        else:
            self.bHaveBackground = False

        self.nameSelected = None
        self.uiSelected = None
        self.stateSelected = None
        self.tagSelected = None
        self.partnameSelected = None
        self.iButtonSelected = None
        self.nameSelectedNow = None
        self.uiSelectedNow = None
        self.fly.update_handle_points()

        self.imgScaled = None
        self.imgUnscaled = None
        self.imgOutput = None
        self.shapeImage = None
        self.header = None

        self.hzF = 0.0
        self.tPrev = None
        self.iCount = 0
        self.dtTrack = 0.0     # Duration of the last fly.update, seconds.

        self.h_gap = int(5 * self.scale)
        self.scaleText = 0.4 * self.scale
        self.fontface = cv2.FONT_HERSHEY_DUPLEX
        self.buttons = None
        self.yToolbar = 0
        self.shapeToolbar = (0, 0)
        self.ibtnInvertColor = None

        if self.use_gui:
            cv2.namedWindow(self.window_name, 1)
            cv2.setMouseCallback(self.window_name, self.on_mouse, param=None)

    # ------------------------------------------------------------------ params

    def legalize_params(self, paramsIn):
        paramsOut = copy.copy(paramsIn)
        for partname in ['head', 'abdomen', 'left', 'right']:
            p = paramsOut['gui'][partname]
            if 'hinge' in p:
                p['hinge']['x'] = max(0, p['hinge']['x'])
                p['hinge']['y'] = max(0, p['hinge']['y'])
            if 'radius_inner' in p:
                p['radius_inner'] = max(5 * paramsOut['scale_image'], p['radius_inner'])
            if 'radius_outer' in p and 'radius_inner' in p:
                p['radius_outer'] = max(p['radius_outer'],
                                        p['radius_inner'] + 5 * paramsOut['scale_image'])
        return paramsOut

    def scale_params(self, paramsIn, scale):
        paramsScaled = copy.deepcopy(paramsIn)
        for partname in ['head', 'abdomen', 'left', 'right']:
            paramsScaled['gui'][partname]['hinge']['x'] = paramsIn['gui'][partname]['hinge']['x'] * scale
            paramsScaled['gui'][partname]['hinge']['y'] = paramsIn['gui'][partname]['hinge']['y'] * scale
            paramsScaled['gui'][partname]['radius_outer'] = paramsIn['gui'][partname]['radius_outer'] * scale
            paramsScaled['gui'][partname]['radius_inner'] = paramsIn['gui'][partname]['radius_inner'] * scale
        for partname in ['aux']:
            paramsScaled['gui'][partname]['center']['x'] = paramsIn['gui'][partname]['center']['x'] * scale
            paramsScaled['gui'][partname]['center']['y'] = paramsIn['gui'][partname]['center']['y'] * scale
            paramsScaled['gui'][partname]['radius1'] = paramsIn['gui'][partname]['radius1'] * scale
            paramsScaled['gui'][partname]['radius2'] = paramsIn['gui'][partname]['radius2'] * scale
        for partname in ['axis']:
            for tag in ['pt1', 'pt2']:
                paramsScaled['gui'][partname][tag]['x'] = paramsIn['gui'][partname][tag]['x'] * scale
                paramsScaled['gui'][partname][tag]['y'] = paramsIn['gui'][partname][tag]['y'] * scale
        return paramsScaled

    def save_params(self, save_reference=True):
        self.config['tracker'] = self.params
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(self.config, f, sort_keys=False)
        # The frame the regions were tuned on becomes the autoplace reference.
        if save_reference and self.imgUnscaled is not None:
            cv2.imwrite(self.filenameReference, self.imgUnscaled)

    # ---------------------------------------------------------------- tracking

    def process(self, frame):
        """Track one Frame; returns a CameraState (or None while the user is
        dragging handles)."""
        with self.lock:
            img = frame.image
            self.imgUnscaled = img
            if self.scale == 1.0:
                self.imgScaled = img
            else:
                self.imgScaled = cv2.resize(img, (0, 0), fx=self.scale, fy=self.scale)
            self.shapeImage = self.imgScaled.shape
            self.bValidImage = True

            # Framerate estimate for the display.
            t = frame.t_host
            if self.tPrev is not None and t > self.tPrev:
                hz = 1.0 / (t - self.tPrev)
                a = 0.04 if self.iCount > 100 else 0.5
                self.hzF = (1 - a) * self.hzF + a * hz
            self.tPrev = t
            self.iCount += 1

            if self.buttons is None or self.imgScaled.shape != self.shapeToolbar:
                self.create_buttons(self.imgScaled.shape)

            if self.bMousing:
                return None

            if self.iCount == 1 and self.config.get('tracker') \
                    and not os.path.exists(self.filenameReference):
                # Existing tuned regions but no reference yet: assume they were
                # tuned on this view and make this frame the autoplace reference.
                cv2.imwrite(self.filenameReference, self.imgUnscaled)
                print('%s: saved first frame as autoplace reference %s'
                      % (self.name, self.filenameReference))
            stamp = frame.t_camera if np.isfinite(frame.t_camera) else frame.t_host
            self.header = Header(seq=frame.frame_id, stamp=stamp, frame_id=self.name)
            t0 = time.monotonic()
            self.fly.update(self.header, self.imgScaled)
            self.dtTrack = time.monotonic() - t0
            msg = self.fly.publish()
            return CameraState.from_flystate_msg(self.name, msg, frame,
                                                 t_done=time.monotonic())

    # ---------------------------------------------------------------- drawing

    def render(self):
        """Draw the tracking overlay; returns the annotated BGR image."""
        with self.lock:
            if self.imgScaled is None:
                return None
            imgOutput = cv2.cvtColor(self.imgScaled, cv2.COLOR_GRAY2BGR)
            self.fly.draw(imgOutput)

            x = int(10 * self.scale)
            y = int(imgOutput.shape[0] - 10 * self.scale)
            h = 10
            h_text = int(h * self.scale)

            if not self.bMousing:
                s = '%5.1fHz  track:%4.1fms' % (self.hzF, self.dtTrack * 1e3)
                cv2.putText(imgOutput, s, (x, y), self.fontface, self.scaleText,
                            ui.bgra_dict['dark_red'])
                if self.statusText:
                    cv2.putText(imgOutput, self.statusText, (x + int(180 * self.scale), y),
                                self.fontface, self.scaleText, ui.bgra_dict['dark_green'])

                if self.params['gui']['aux']['track']:
                    y -= h_text + self.h_gap
                    s = 'AUX: (%0.3f)' % self.fly.aux.state.intensity
                    cv2.putText(imgOutput, s, (x, y), self.fontface, self.scaleText, self.fly.aux.bgra)
                    y -= h_text + self.h_gap
                    s = 'WB Freq: '
                    if self.fly.aux.state.freq != 0.0:
                        s += '%0.0fhz' % self.fly.aux.state.freq
                    cv2.putText(imgOutput, s, (x, y), self.fontface, self.scaleText, self.fly.aux.bgra)

                if self.params['gui']['left']['track'] and self.params['gui']['right']['track']:
                    if len(self.fly.left.state.angles) > 0 and len(self.fly.right.state.angles) > 0:
                        L = self.fly.left.state.angles[0]
                        R = self.fly.right.state.angles[0]
                        y -= h_text + self.h_gap
                        cv2.putText(imgOutput, 'L+R:% 7.4f' % (L + R), (x, y),
                                    self.fontface, self.scaleText, ui.bgra_dict['blue'])
                        y -= h_text + self.h_gap
                        cv2.putText(imgOutput, 'L-R:% 7.4f' % (L - R), (x, y),
                                    self.fontface, self.scaleText, ui.bgra_dict['blue'])

                for partname, label in (('right', 'R'), ('left', 'L'),
                                        ('abdomen', 'ABDOMEN'), ('head', 'HEAD')):
                    if self.params['gui'][partname]['track']:
                        part = getattr(self.fly, partname)
                        y -= h_text + self.h_gap
                        s = label + ':'
                        if len(part.state.angles) > 0:
                            s += '% 7.4f' % part.state.angles[0]
                        cv2.putText(imgOutput, s, (x, y), self.fontface, self.scaleText, part.bgra)

            if self.ibtnInvertColor is not None:
                self.buttons[self.ibtnInvertColor].state = self.fly.bInvertColor
            for btn in self.buttons or []:
                btn.draw(imgOutput)

            self.imgOutput = imgOutput
            return imgOutput

    # ---------------------------------------------------------------- buttons

    def wrap_button(self, btn, shape):
        if btn.right >= shape[1]:
            btn.set_pos(pt=[1, btn.bottom + 1])

    def create_buttons(self, shape):
        self.shapeToolbar = shape
        self.buttons = []

        def add(**kwargs):
            if self.buttons:
                prev = self.buttons[-1]
                pt = [prev.right + 1, prev.top + 1]
            else:
                pt = [1, 1]
            btn = ui.Button(pt=pt, scale=self.scale, **kwargs)
            self.wrap_button(btn, shape)
            self.buttons.append(btn)
            return btn

        add(type='pushbutton', name='exit', text='exit')
        add(type='pushbutton', name='save_bg', text='saveBG')
        add(type='pushbutton', name='autoplace', text='autoplace')
        add(type='pushbutton', name='reset', text='reset')
        add(type='static', name='track', text='track:',
            sides=ui.SIDE_LEFT | ui.SIDE_TOP | ui.SIDE_BOTTOM)
        for partname, label in (('head', 'H'), ('abdomen', 'A'), ('left', 'L'),
                                ('right', 'R')):
            add(type='checkbox', name=partname, text=label,
                state=self.params['gui'][partname]['track'],
                sides=ui.SIDE_TOP | ui.SIDE_BOTTOM)
        add(type='checkbox', name='aux', text='X',
            state=self.params['gui']['aux']['track'],
            sides=ui.SIDE_TOP | ui.SIDE_BOTTOM | ui.SIDE_RIGHT)
        add(type='static', name='subtract', text='subtract:',
            sides=ui.SIDE_LEFT | ui.SIDE_TOP | ui.SIDE_BOTTOM)
        add(type='checkbox', name='subtract_head', text='H',
            state=self.params['gui']['head']['subtract_bg'],
            sides=ui.SIDE_TOP | ui.SIDE_BOTTOM)
        add(type='checkbox', name='subtract_abdomen', text='A',
            state=self.params['gui']['abdomen']['subtract_bg'],
            sides=ui.SIDE_TOP | ui.SIDE_BOTTOM)
        add(type='checkbox', name='subtract_lr', text='LR',
            state=self.params['gui']['right']['subtract_bg'],
            sides=ui.SIDE_TOP | ui.SIDE_BOTTOM)
        add(type='checkbox', name='subtract_aux', text='X',
            state=self.params['gui']['aux']['subtract_bg'],
            sides=ui.SIDE_TOP | ui.SIDE_BOTTOM | ui.SIDE_RIGHT)
        add(type='checkbox', name='invertcolor', text='invertcolor',
            state=self.fly.bInvertColor)
        self.ibtnInvertColor = len(self.buttons) - 1
        add(type='checkbox', name='stabilize', text='stabilize',
            state=self.params['gui']['head']['stabilize'])
        add(type='checkbox', name='symmetry', text='symmetric',
            state=self.params['gui']['symmetric'])
        add(type='checkbox', name='windows', text='windows',
            state=self.params['gui']['windows'])
        self.yToolbar = self.buttons[-1].bottom + 1

    # ------------------------------------------------------------------ mouse

    def save_background(self):
        print('%s: saving new background image to %s' % (self.name, self.filenameBackground))
        cv2.imwrite(self.filenameBackground, self.imgUnscaled)
        self.fly.set_background(self.imgScaled)
        self.bHaveBackground = True

    def apply_params(self, params, status, save=True):
        """Install a new params dict (from autoplace/reset), rebuild masks,
        and persist it. Points are clamped into the image first so handles
        can never end up outside the field of view."""
        params = autoplace.clamp_to_image(params, self.imgUnscaled.shape)
        self.params = self.legalize_params(params)
        for part in (self.fly.axis, self.fly.head, self.fly.abdomen,
                     self.fly.left, self.fly.right, self.fly.aux):
            part.bValidMask = False
        self.fly.set_params(self.scale_params(self.params, self.scale))
        self.fly.create_masks(self.shapeImage)
        self.fly.update_handle_points()
        self.statusText = status
        print('%s: %s' % (self.name, status))
        if save:
            self.save_params(save_reference=False)

    def reset_params(self):
        """Undo the last autoplace; a second press (or with nothing to undo)
        restores the layout loaded at startup."""
        if self.imgUnscaled is None:
            return
        if self.paramsUndo is not None:
            params, self.paramsUndo = self.paramsUndo, None
            self.apply_params(params, 'reset: undid autoplace')
        else:
            self.apply_params(copy.deepcopy(self.paramsStartup), 'reset: startup layout restored')

    def autoplace(self):
        """Place the tracking regions from the current image by registering
        it to the saved reference frame. The result is applied only if the
        transform is plausible and keeps every region inside the image;
        otherwise nothing changes. 'reset' undoes it."""
        if self.imgUnscaled is None:
            return
        ref = cv2.imread(self.filenameReference, cv2.IMREAD_GRAYSCALE)
        if ref is None or ref.shape != self.imgUnscaled.shape:
            self.statusText = 'autoplace: no reference frame yet - adjust any handle once to save one'
            print('%s: %s' % (self.name, self.statusText))
            return
        try:
            mask = autoplace.body_mask(self.params, ref.shape)
            W, method = autoplace.register(ref, self.imgUnscaled, mask)
            shift = (float(W[0, 2]), float(W[1, 2]))
            rot = np.degrees(np.arctan2(W[1, 0], W[0, 0]))
            if not autoplace.plausible_transform(W, ref.shape):
                self.statusText = ('autoplace: implausible (shift %.0f,%.0f px, rot %.0f deg) - unchanged'
                                   % (shift[0], shift[1], rot))
                print('%s: %s' % (self.name, self.statusText))
                return
            moved, _ = autoplace.transform_params(self.params, W)
            if not autoplace.points_inside(moved, ref.shape):
                self.statusText = 'autoplace: would move regions out of the image - unchanged'
                print('%s: %s' % (self.name, self.statusText))
                return
        except Exception as e:
            self.statusText = 'autoplace failed: %s' % e
            print('%s: %s' % (self.name, self.statusText))
            return
        # Keep a backup of the YAML and an in-memory undo point.
        try:
            import shutil
            shutil.copy2(self.config_path, self.config_path + '.bak')
        except OSError:
            pass
        self.paramsUndo = copy.deepcopy(self.params)
        self.apply_params(moved, 'autoplace: %s  shift (%.1f, %.1f) px  rot %.1f deg  [reset = undo]'
                          % (method, shift[0], shift[1], rot))

    def hit_object(self, ptMouse):
        tagHit = None
        partnameHit = None
        uiHit = None
        iButtonHit = None
        for iButton in range(len(self.buttons)):
            if self.buttons[iButton].hit_test(ptMouse):
                iButtonHit = iButton
        if iButtonHit is not None:
            nameNearest = self.buttons[iButtonHit].name
            (tagHit, delim, partnameHit) = nameNearest.partition('_')
            uiHit = self.buttons[iButtonHit].type
        else:
            hits = [self.fly.left.hit_object(ptMouse),
                    self.fly.right.hit_object(ptMouse),
                    self.fly.head.hit_object(ptMouse),
                    self.fly.abdomen.hit_object(ptMouse),
                    self.fly.aux.hit_object(ptMouse),
                    self.fly.axis.hit_object(ptMouse)]
            for (partname, tag) in hits:
                if tag is not None:
                    tagHit = tag
                    partnameHit = partname
                    uiHit = 'handle'
                    break
        return (uiHit, tagHit, partnameHit, iButtonHit)

    def name_from_tagpartname(self, tag, partname):
        if partname is not None and len(partname) > 0:
            return tag + '_' + partname
        return tag

    def bodypart_from_partname(self, partname):
        return {'left': self.fly.left, 'right': self.fly.right,
                'head': self.fly.head, 'abdomen': self.fly.abdomen,
                'aux': self.fly.aux, 'axis': self.fly.axis}.get(partname)

    def update_params_from_mouse(self, tagSelected, partnameSelected, ptMouse):
        partnameSlave = {'axis': None, 'head': 'abdomen', 'abdomen': 'head',
                         'right': 'left', 'left': 'right', 'aux': None}[partnameSelected]
        tagOther = 'angle_lo' if (tagSelected == 'angle_hi') else 'angle_hi'
        tagSlave = tagOther
        bodypartSelected = self.bodypart_from_partname(partnameSelected)
        bodypartSlave = self.bodypart_from_partname(partnameSlave)

        paramsScaled = self.scale_params(self.params, self.scale)

        if tagSelected == 'hinge':
            if partnameSelected in ('head', 'abdomen'):
                if paramsScaled['gui']['symmetric']:
                    ptHead = np.array([paramsScaled['gui']['head']['hinge']['x'],
                                       paramsScaled['gui']['head']['hinge']['y']])
                    ptAbdomen = np.array([paramsScaled['gui']['abdomen']['hinge']['x'],
                                          paramsScaled['gui']['abdomen']['hinge']['y']])
                    ptCenterPre = (ptHead + ptAbdomen) / 2
                    ptBodyPre = ptHead - ptAbdomen
                    angleBodyPre = np.arctan2(ptBodyPre[1], ptBodyPre[0])
                    ptLeft = np.array([paramsScaled['gui']['left']['hinge']['x'],
                                       paramsScaled['gui']['left']['hinge']['y']])
                    ptRight = np.array([paramsScaled['gui']['right']['hinge']['x'],
                                        paramsScaled['gui']['right']['hinge']['y']])
                    ptLC = ptLeft - ptCenterPre
                    ptRC = ptRight - ptCenterPre
                    rL = np.linalg.norm(ptLC)
                    aL = np.arctan2(ptLC[1], ptLC[0]) - angleBodyPre
                    rR = np.linalg.norm(ptRC)
                    aR = np.arctan2(ptRC[1], ptRC[0]) - angleBodyPre

                paramsScaled['gui'][partnameSelected]['hinge']['x'] = float(ptMouse[0])
                paramsScaled['gui'][partnameSelected]['hinge']['y'] = float(ptMouse[1])

                for part in (self.fly.axis, self.fly.head, self.fly.abdomen,
                             self.fly.left, self.fly.right):
                    part.bValidMask = False

                if paramsScaled['gui']['symmetric']:
                    ptHead = np.array([paramsScaled['gui']['head']['hinge']['x'],
                                       paramsScaled['gui']['head']['hinge']['y']])
                    ptAbdomen = np.array([paramsScaled['gui']['abdomen']['hinge']['x'],
                                          paramsScaled['gui']['abdomen']['hinge']['y']])
                    ptCenterPost = (ptHead + ptAbdomen) / 2
                    ptBodyPost = ptHead - ptAbdomen
                    angleBodyPost = np.arctan2(ptBodyPost[1], ptBodyPost[0])
                    ptLeft = ptCenterPost + rL * np.array([np.cos(aL + angleBodyPost),
                                                           np.sin(aL + angleBodyPost)])
                    ptRight = ptCenterPost + rR * np.array([np.cos(aR + angleBodyPost),
                                                            np.sin(aR + angleBodyPost)])
                    paramsScaled['gui']['left']['hinge']['x'] = float(ptLeft[0])
                    paramsScaled['gui']['left']['hinge']['y'] = float(ptLeft[1])
                    paramsScaled['gui']['right']['hinge']['x'] = float(ptRight[0])
                    paramsScaled['gui']['right']['hinge']['y'] = float(ptRight[1])

            elif partnameSelected in ('left', 'right'):
                paramsScaled['gui'][partnameSelected]['hinge']['x'] = float(ptMouse[0])
                paramsScaled['gui'][partnameSelected]['hinge']['y'] = float(ptMouse[1])
                bodypartSelected.bValidMask = False
                if paramsScaled['gui']['symmetric']:
                    ptSlave = imageprocessing.get_reflection_across_axis(
                        ptMouse, (self.fly.abdomen.ptHinge_i, self.fly.head.ptHinge_i))
                    paramsScaled['gui'][partnameSlave]['hinge']['x'] = float(ptSlave[0])
                    paramsScaled['gui'][partnameSlave]['hinge']['y'] = float(ptSlave[1])
                    bodypartSlave.bValidMask = False

        elif tagSelected in ('angle_hi', 'angle_lo'):
            pt = ptMouse - bodypartSelected.ptHinge_i
            if tagSelected == 'angle_lo':
                angle_lo_b = float(bodypartSelected.transform_angle_b_from_i(np.arctan2(pt[1], pt[0])))
                if partnameSelected in ('head', 'abdomen') and paramsScaled['gui']['symmetric']:
                    angle_hi_b = -angle_lo_b
                else:
                    angle_hi_b = paramsScaled['gui'][partnameSelected][tagOther]
            else:
                angle_hi_b = float(bodypartSelected.transform_angle_b_from_i(np.arctan2(pt[1], pt[0])))
                if partnameSelected in ('head', 'abdomen') and paramsScaled['gui']['symmetric']:
                    angle_lo_b = -angle_hi_b
                else:
                    angle_lo_b = paramsScaled['gui'][partnameSelected][tagOther]

            paramsScaled['gui'][partnameSelected]['radius_outer'] = float(
                max(bodypartSelected.params['gui'][partnameSelected]['radius_inner'] + 2 * self.scale,
                    np.linalg.norm(bodypartSelected.ptHinge_i - ptMouse)))

            angle_lo_b -= bodypartSelected.angleBodypart_b
            angle_lo_b = (angle_lo_b + np.pi) % (2.0 * np.pi) - np.pi
            angle_hi_b -= bodypartSelected.angleBodypart_b
            angle_hi_b = (angle_hi_b + np.pi) % (2.0 * np.pi) - np.pi

            if not (angle_lo_b < angle_hi_b):
                self.tagSelected = tagOther

            paramsScaled['gui'][partnameSelected]['angle_lo'] = min(angle_lo_b, angle_hi_b)
            paramsScaled['gui'][partnameSelected]['angle_hi'] = max(angle_lo_b, angle_hi_b)
            paramsScaled['gui'][partnameSelected]['angle_lo'] += bodypartSelected.angleBodypart_b
            paramsScaled['gui'][partnameSelected]['angle_hi'] += bodypartSelected.angleBodypart_b
            paramsScaled['gui'][partnameSelected]['angle_lo'] = \
                (paramsScaled['gui'][partnameSelected]['angle_lo'] + np.pi) % (2.0 * np.pi) - np.pi
            paramsScaled['gui'][partnameSelected]['angle_hi'] = \
                (paramsScaled['gui'][partnameSelected]['angle_hi'] + np.pi) % (2.0 * np.pi) - np.pi
            if paramsScaled['gui'][partnameSelected]['angle_hi'] < paramsScaled['gui'][partnameSelected]['angle_lo']:
                paramsScaled['gui'][partnameSelected]['angle_hi'] += 2 * np.pi

            bodypartSelected.bValidMask = False

            if partnameSelected in ('left', 'right') and paramsScaled['gui']['symmetric']:
                paramsScaled['gui'][partnameSlave][tagSlave] = -paramsScaled['gui'][partnameSelected][tagSelected]
                paramsScaled['gui'][partnameSlave][tagSelected] = -paramsScaled['gui'][partnameSelected][tagSlave]
                paramsScaled['gui'][partnameSlave]['radius_outer'] = paramsScaled['gui'][partnameSelected]['radius_outer']
                paramsScaled['gui'][partnameSlave]['radius_inner'] = paramsScaled['gui'][partnameSelected]['radius_inner']
                bodypartSlave.bValidMask = False

        elif tagSelected == 'radius_inner':
            paramsScaled['gui'][partnameSelected]['radius_inner'] = float(
                min(np.linalg.norm(bodypartSelected.ptHinge_i - ptMouse),
                    bodypartSelected.params['gui'][partnameSelected]['radius_outer'] - 2 * self.scale))
            bodypartSelected.bValidMask = False
            if partnameSelected in ('left', 'right') and paramsScaled['gui']['symmetric']:
                paramsScaled['gui'][partnameSlave]['radius_outer'] = paramsScaled['gui'][partnameSelected]['radius_outer']
                paramsScaled['gui'][partnameSlave]['radius_inner'] = paramsScaled['gui'][partnameSelected]['radius_inner']
                bodypartSlave.bValidMask = False

        elif tagSelected == 'center':
            if partnameSelected == 'aux':
                paramsScaled['gui'][partnameSelected]['center']['x'] = float(ptMouse[0])
                paramsScaled['gui'][partnameSelected]['center']['y'] = float(ptMouse[1])
                bodypartSelected.bValidMask = False

        elif tagSelected in ('pt1', 'pt2'):
            if partnameSelected == 'axis':
                paramsScaled['gui'][partnameSelected][tagSelected]['x'] = float(ptMouse[0])
                paramsScaled['gui'][partnameSelected][tagSelected]['y'] = float(ptMouse[1])
                bodypartSelected.bValidMask = False

        elif tagSelected == 'radius1':
            pt = ptMouse - bodypartSelected.ptCenter_i
            paramsScaled['gui'][partnameSelected]['radius1'] = float(np.linalg.norm(pt))
            paramsScaled['gui'][partnameSelected]['angle'] = float(np.arctan2(pt[1], pt[0]))
            bodypartSelected.bValidMask = False
        elif tagSelected == 'radius2':
            pt = bodypartSelected.ptCenter_i - ptMouse
            paramsScaled['gui'][partnameSelected]['radius2'] = float(np.linalg.norm(pt))
            paramsScaled['gui'][partnameSelected]['angle'] = float(np.arctan2(pt[1], pt[0]) - np.pi / 2.0)
            bodypartSelected.bValidMask = False

        self.params = self.scale_params(paramsScaled, 1 / self.scale)

    def on_mouse(self, event, x, y, flags, param):
        with self.lock:
            self._on_mouse(event, x, y, flags, param)

    def _on_mouse(self, event, x, y, flags, param):
        if self.shapeImage is None or self.buttons is None:
            return
        ptMouse = np.array([x, y]).clip((0, 0), (self.shapeImage[1], self.shapeImage[0]))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.bMousing = True
            (uiHit, tag, partname, iButtonSelected) = self.hit_object(ptMouse)
            self.tagSelected = tag
            self.partnameSelected = partname
            self.uiSelected = uiHit
            self.iButtonSelected = iButtonSelected
            self.nameSelected = self.name_from_tagpartname(tag, partname)
            if self.iButtonSelected is not None:
                self.stateSelected = self.buttons[self.iButtonSelected].state
            self.nameSelectedNow = self.nameSelected
            self.uiSelectedNow = self.uiSelected

        if self.uiSelected in ('pushbutton', 'checkbox'):
            (uiHit, tag, partname, iButtonSelectedNow) = self.hit_object(ptMouse)
            self.nameSelectedNow = self.name_from_tagpartname(tag, partname)
            self.iButtonSelectedNow = iButtonSelectedNow

            for iButton in range(len(self.buttons)):
                if self.buttons[iButton].type == 'pushbutton':
                    if (iButton == self.iButtonSelectedNow == self.iButtonSelected) \
                            and not (event == cv2.EVENT_LBUTTONUP):
                        self.buttons[iButton].state = True
                    else:
                        self.buttons[iButton].state = False

            if self.uiSelected == 'checkbox':
                if self.nameSelected == self.nameSelectedNow:
                    self.buttons[self.iButtonSelected].state = not self.stateSelected
                else:
                    self.buttons[self.iButtonSelected].state = self.stateSelected

        elif self.uiSelected == 'handle':
            self.update_params_from_mouse(
                self.tagSelected, self.partnameSelected,
                ptMouse.clip((0, self.yToolbar), (self.shapeImage[1], self.shapeImage[0])))
            self.fly.set_params(self.scale_params(self.params, self.scale))

        if event == cv2.EVENT_LBUTTONUP:
            iButtonSelected = self.iButtonSelected
            if self.uiSelected == 'pushbutton':
                if self.nameSelected == self.nameSelectedNow == 'save_bg':
                    self.save_background()
                elif self.nameSelected == self.nameSelectedNow == 'exit':
                    self.bQuit = True
                elif self.nameSelected == self.nameSelectedNow == 'autoplace':
                    self.autoplace()
                elif self.nameSelected == self.nameSelectedNow == 'reset':
                    self.reset_params()

            elif self.uiSelected == 'checkbox':
                if self.nameSelected == self.nameSelectedNow:
                    self.buttons[self.iButtonSelected].state = not self.stateSelected
                else:
                    self.buttons[self.iButtonSelected].state = self.stateSelected
                state = self.buttons[iButtonSelected].state
                name = self.nameSelected if (self.nameSelected == self.nameSelectedNow) else None

                if name == 'symmetry':
                    self.params['gui']['symmetric'] = state
                elif name == 'subtract_lr':
                    if not self.bHaveBackground:
                        self.buttons[iButtonSelected].state = state = False
                        print('%s: no background image; cannot subtract for left/right.' % self.name)
                    self.params['gui']['left']['subtract_bg'] = state
                    self.params['gui']['right']['subtract_bg'] = state
                elif name == 'subtract_aux':
                    if not self.bHaveBackground:
                        self.buttons[iButtonSelected].state = state = False
                        print('%s: no background image; cannot subtract for aux.' % self.name)
                    self.params['gui']['aux']['subtract_bg'] = state
                elif name == 'subtract_head':
                    if not self.bHaveBackground:
                        self.buttons[iButtonSelected].state = state = False
                        print('%s: no background image; cannot subtract for head.' % self.name)
                    self.params['gui']['head']['subtract_bg'] = state
                elif name == 'subtract_abdomen':
                    if not self.bHaveBackground:
                        self.buttons[iButtonSelected].state = state = False
                        print('%s: no background image; cannot subtract for abdomen.' % self.name)
                    self.params['gui']['abdomen']['subtract_bg'] = state
                elif name in ('head', 'abdomen', 'left', 'right', 'aux'):
                    self.params['gui'][name]['track'] = state
                elif name == 'invertcolor':
                    self.fly.bInvertColor = state
                    self.fly.bInvertColorValid = True
                    self.fly.bInvertColorAuto = False
                    self.params['invertcolor'] = bool(state)
                elif name == 'stabilize':
                    for partname in ('head', 'abdomen', 'left', 'right'):
                        self.params['gui'][partname]['stabilize'] = state
                elif name == 'windows':
                    self.params['gui']['windows'] = state

            if self.uiSelected in ('handle', 'checkbox'):
                self.fly.set_params(self.scale_params(self.params, self.scale))
                self.fly.create_masks(self.shapeImage)
            self.save_params()

            self.bMousing = False
            self.nameSelected = None
            self.nameSelectedNow = None
            self.uiSelected = None
            self.uiSelectedNow = None
            self.iButtonSelected = None
