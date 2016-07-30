#!/usr/bin/env python2
from __future__ import \
    ( nested_scopes
    , generators
    , division
    , absolute_import
    , with_statement
    , print_function
    , unicode_literals
    )

import time

import cv2
import numpy

events = \
    { getattr(cv2, attr): attr
      for attr in dir(cv2)
      if attr.startswith('EVENT')
      and not attr.startswith('EVENT_FLAG')
    }

class MouseQuery(object):

    def __init__ \
            ( self
            , open_window
            , image
            , point_count = 1
            , annot_fn = lambda im, lmb, xy, pts: None
            ):
        self._open_window = open_window
        self._image = image
        self._point_count_target = point_count
        self._user_annotation = annot_fn
        # TODO: replace with namedtuple
        self.__dst_image = numpy.empty_like(image)
        self.__mouse_is_down = False
        self.__current_mouse_loc = (0, 0)
        self.__points_collected = []

    def __enter__(self):
        self._open_window.on_mouse(self._on_mouse)
        return self._loop

    def __exit__(self, exc_type, exc_value, traceback):
        self._open_window.on_mouse(None)

    def _on_mouse(self, event, x, y, flags, param):
        if events[event] == 'EVENT_MOUSEMOVE':
            self.__current_mouse_loc = (x, y)
        elif events[event] == 'EVENT_LBUTTONDOWN':
            self.__mouse_is_down = True
        elif events[event] == 'EVENT_LBUTTONUP' and self.__mouse_is_down:
            self.__mouse_is_down = False
            self.__points_collected.append((x, y))
        else:
            print('Ignoring:', events[event], x, y, flags, param)

    def _loop(self):
        while len(self.__points_collected) < self._point_count_target:
            numpy.copyto(self.__dst_image, self._image)
            self._user_annotation(self.__dst_image, self.__mouse_is_down, self.__current_mouse_loc, self.__points_collected)
            if self._open_window.ims_show(self.__dst_image) == 27:
                raise ValueError('ESC key')
        return self.__points_collected

pulse_brightness = lambda: int(time.time() * 2) % 2 * 255
pulse = lambda dst, _, __, ___: (pulse_brightness(),) * (1 if dst.ndim <= 2 else dst.shape[2])

def annotate_box(dst, lmb, xy, pts, color_fn=None):
    if pts:
        c = (color_fn or pulse)(dst, lmb, xy, pts)
        cv2.rectangle(dst, pts[-1], xy, c)

def annotate_quadrants(dst, lmb, xy, pts, color_fn=None):
    c = (color_fn or pulse)(dst, lmb, xy, pts)
    ym, xm = dst.shape[:2]
    x, y = xy
    cv2.line(dst, (0, y), (xm, y), c)
    cv2.line(dst, (x, 0), (x, ym), c)

def annotate_reticle(dst, lmb, xy, pts, color_fn=None, size=None, thickness=None):
    t = 1 if thickness is None else thickness
    s = 4 if size is None else size
    c = (color_fn or pulse)(dst, lmb, xy, pts)
    x, y = xy
    cv2.line(dst, (x - s, y), (x + s, y), c, t)
    cv2.line(dst, (x, y - s), (x, y + s), c, t)

# eof
