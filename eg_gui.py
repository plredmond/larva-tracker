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

import sys
import time
import itertools
import functools

import cv2
import numpy

import lib.cvutils as cvutils

events = \
    { getattr(cv2, attr): attr
      for attr in dir(cv2)
      if attr.startswith('EVENT')
      and not attr.startswith('EVENT_FLAG')
    }

class MouseQuery(object):

    def __init__ \
            ( self
            , window
            , image
            , point_count_target = 1
            , reticle_size = 4
            , user_annotation = lambda im, xy, pts: None
            , reticle_color = (0, 0, 255)
            , big_reticle_color = None
            ):
        self._window = window
        self._image = image
        self._point_count_target = point_count_target
        self._reticle_size = reticle_size
        self._user_annotation = user_annotation
        self._reticle_color = reticle_color
        self._big_reticle_color = big_reticle_color
        self.__dst_image = numpy.empty_like(image)
        self.__mouse_is_down = False
        self.__current_mouse_loc = (0, 0)
        self.__points_collected = []

    def __enter__(self):
        cv2.setMouseCallback(self._window, self._onMouse)
        return self._loop

    def __exit__(self, exc_type, exc_value, traceback):
        cv2.setMouseCallback(self._window, lambda *_: None)

    def _onMouse(self, event, x, y, flags, param):
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
            # blank
            numpy.copyto(self.__dst_image, self._image)
            # user annotation
            self._user_annotation(self.__dst_image, self.__current_mouse_loc, self.__points_collected)
            # reticle
            x, y = self.__current_mouse_loc
            if self._big_reticle_color is not None and self.__mouse_is_down:
                ym, xm = self.__dst_image.shape[:2]
                cv2.line(self.__dst_image, (0, y), (xm, y), self._big_reticle_color)
                cv2.line(self.__dst_image, (x, 0), (x, ym), self._big_reticle_color)
            if self._reticle_color:
                cv2.line(self.__dst_image, (x - self._reticle_size, y), (x + self._reticle_size, y), self._reticle_color)
                cv2.line(self.__dst_image, (x, y - self._reticle_size), (x, y + self._reticle_size), self._reticle_color)
            # display
            cv2.imshow(w, self.__dst_image)
            if cv2.waitKey(1) == 27:
                raise ValueError('ESC key')
        return self.__points_collected

if __name__ == '__main__':
    m = cvutils.Capture.argtype(sys.argv[1] if sys.argv[1:] else 0)
    w = 'opencv'
    cv2.namedWindow(w, cv2.WINDOW_NORMAL)

    first = next(m)
    def a(dst, xy, pts):
        c = time.time() % 1 * 255
        assert 0 <= c < 255
        if pts:
            cv2.rectangle(dst, pts[0], xy, [c] * 3)
    with MouseQuery(w, first, count=2, size=20, annot=a, big_reticle_color=(0,255,255)) as loop:
        box = loop()

    for frame in itertools.chain([first], m):
        a(frame, box[1], box)
        cv2.imshow(w, frame)
        cv2.waitKey(1) == 27 and exit(0)

# eof
