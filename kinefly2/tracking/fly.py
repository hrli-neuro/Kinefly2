#!/usr/bin/env python

import cv2
import numpy as np

from .imagewindow import ImageWindow
from .tracker_axis import AxisTracker
from . import parts as partspec
from . import ui
from . import imageprocessing

from .MsgFlyState import MsgFlyState, MsgState , Header


###############################################################################
###############################################################################
class Fly(object):
    def __init__(self, params={}):
        #self.nodename = rospy.get_name().rstrip('/')
        self.nodename = 'kinefly'

        # Create the body axis tracker.
        self.axis        = AxisTracker(name='axis',      params=params, color='cyan')

        # Every part, built from what its config asks for. The names come from
        # tracking/parts.py, so a camera that declares extra parts (the hind
        # legs on the ventral view) gets them here without a code change.
        self.partnames = partspec.body_parts(params)
        self.parts = {}
        for name in partspec.all_parts(params):
            try:
                tracker = partspec.make_tracker(name, params)
            except ValueError as e:
                print(e)
                continue
            self.parts[name] = tracker
            # fly.head, fly.left, ... still work; extras are reachable the
            # same way (getattr(fly, 'leg_r3')).
            setattr(self, name, tracker)

        self.windowInvertColorArea      = ImageWindow(False, 'InvertColorArea')
        
        self.bgra_body = ui.bgra_dict['light_gray']
        self.ptBodyIndicator1 = None
        self.ptBodyIndicator2 = None
        self.bInvertColor = False
        self.bInvertColorAuto = True
        self.iCount  = 0
        self.stampPrev = None
        self.stampPrevAlt = None

        self.stamp = 0

    def set_params(self, params):
        # The body frame the wing angles are reported in. Deriving it from the
        # head/abdomen hinges on every call couples those parts to the wings:
        # move either hinge and every wing angle shifts. Freeze it in the
        # config instead, seeded once from the hinges so existing recordings
        # stay comparable. Must happen before the parts read it.
        if params.get('body_axis_deg') is None:
            params['body_axis_deg'] = float(np.degrees(
                imageprocessing.get_angle_from_points_i(
                    np.array([params['gui']['abdomen']['hinge']['x'],
                              params['gui']['abdomen']['hinge']['y']]),
                    np.array([params['gui']['head']['hinge']['x'],
                              params['gui']['head']['hinge']['y']]))))
        self.params = params
        
        self.axis.set_params(params)
        for part in self.parts.values():
            part.set_params(params)

        pt1 = [params['gui']['head']['hinge']['x'], params['gui']['head']['hinge']['y']]
        pt2 = [params['gui']['abdomen']['hinge']['x'], params['gui']['abdomen']['hinge']['y']]
        pt3 = [params['gui']['left']['hinge']['x'], params['gui']['left']['hinge']['y']]
        pt4 = [params['gui']['right']['hinge']['x'], params['gui']['right']['hinge']['y']]
        self.ptBodyCenter_i = imageprocessing.get_intersection(pt1,pt2,pt3,pt4)

        r = max(params['gui']['left']['radius_outer'], params['gui']['right']['radius_outer'])
        self.angleBody_i = float(np.radians(params['body_axis_deg']))
        self.ptBodyIndicator1 = tuple((self.ptBodyCenter_i + r * np.array([np.cos(self.angleBody_i), np.sin(self.angleBody_i)])).astype(int))
        self.ptBodyIndicator2 = tuple((self.ptBodyCenter_i - r * np.array([np.cos(self.angleBody_i), np.sin(self.angleBody_i)])).astype(int))
        
        # Radius of an area approximately where the thorax would be.
        self.rInvertColorArea = np.linalg.norm(np.array([params['gui']['head']['hinge']['x'], params['gui']['head']['hinge']['y']]) - np.array([params['gui']['abdomen']['hinge']['x'], params['gui']['abdomen']['hinge']['y']]))/2.0
        self.bInvertColorValid = False
        
    
    def create_masks(self, shapeImage):
        for name, part in [('axis', self.axis)] + list(self.parts.items()):
            if self.params['gui'][name]['track'] and not part.bValidMask:
                part.create_mask(shapeImage)
                part.bValidMask = True


    def get_bodyangle_i(self):
        angle_i = imageprocessing.get_angle_from_points_i(self.abdomen.ptHinge_i, self.head.ptHinge_i)
        #angleBody_i  = (angle_i + np.pi) % (2.0*np.pi) - np.pi
        angleBody_i  = angle_i
        
        return angleBody_i
        

    # Calculate what we think the bInvertColor flag should be to make white-on-black.        
    def get_invertcolor(self, image):
        # Get a roi around the body center.
        xMin = int(max(0,self.ptBodyCenter_i[0]-int(0.75*self.rInvertColorArea)))
        yMin = int(max(0,self.ptBodyCenter_i[1]-int(0.75*self.rInvertColorArea)))
        xMax = int(min(self.ptBodyCenter_i[0]+int(0.75*self.rInvertColorArea), image.shape[1]-1))
        yMax = int(min(self.ptBodyCenter_i[1]+int(0.75*self.rInvertColorArea), image.shape[0]-1))
        imgInvertColorArea = image[yMin:yMax, xMin:xMax]
        self.windowInvertColorArea.set_image(imgInvertColorArea)

        # Midpoint between darkest & lightest colors.
        threshold = np.mean(image) 
        #rospy.logwarn((np.min(image), np.median(image), np.mean(image), np.max(image), np.mean(imgInvertColorArea)))
        
        # If the roi is too dark, then set bInvertColor.
        if (np.mean(imgInvertColorArea) <= threshold):
            bInvertColor = True
        else:
            bInvertColor = False

        return bInvertColor
        
        
                
    def set_background(self, image):
        for part in self.parts.values():
            part.set_background(image)
        self.axis.set_background(image)

    
    def update_handle_points(self):
        for part in self.parts.values():
            part.update_handle_points()
        self.axis.update_handle_points()
        

    def update(self, header=None, image=None):
        if (image is not None):
            self.header = header
            
            if (not self.bInvertColorValid) and (self.bInvertColorAuto):
                self.bInvertColor = self.get_invertcolor(image)
                self.bInvertColorValid = True

            
            # Get the dt from the header timestamps (seconds).  The background-model
            # time constant and the wingbeat detector both depend on real dt values.
            self.stamp = self.header.stamp

            if (self.stampPrev is not None):
                dt = max(0.0, self.stamp - self.stampPrev)

                # If the source is not giving good timestamps, then just assume a value.
                if (dt == 0.0):
                    dt = 1.0

            else:
                dt = np.inf

            self.stampPrev = self.stamp

            for part in self.parts.values():
                part.update(dt, image, self.bInvertColor)
            self.axis.update(dt, image, self.bInvertColor)
            
            
    def draw(self, image):
        # The body-axis line is the frame the wing and abdomen angles are
        # reported in, so it is only meaningful when one of those is tracked.
        # On a frontal camera tracking just the head it means nothing.
        framed = [p for p in self.partnames if p != 'head']
        if any(self.params['gui'][p]['track'] for p in framed):
            cv2.line(image, self.ptBodyIndicator1, self.ptBodyIndicator2, self.bgra_body, 1)

        self.axis.draw(image)
        for part in self.parts.values():
            part.draw(image)

        self.windowInvertColorArea.show()
        
    
    def publish(self):
        flystate              = MsgFlyState()
        flystate.header       = Header(seq=self.iCount, stamp=self.stamp, frame_id='Fly')

        # An untracked part publishes an empty state rather than a stale one,
        # so a part switched off mid-session goes to NaN instead of freezing.
        flystate.partnames = tuple(self.parts)
        for name, part in self.parts.items():
            setattr(flystate, name,
                    part.state if self.params['gui'][name]['track'] else MsgState())

        self.iCount += 1

        return flystate
        


# end class Fly