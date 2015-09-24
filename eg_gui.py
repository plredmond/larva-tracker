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

    def __init__(self, window, image, count=None, size=None, annot=None):
        self.__w = window
        self.__src = image
        self.__dst = numpy.empty_like(image)
        self.__ct = 1 if count is None else count
        self.__s = 4 if size is None else size
        self.__an = (lambda im, pts: None) if annot is None else annot
        self.__xy = (0, 0)
        self.__pts = []

    def __enter__(self):
        cv2.setMouseCallback(self.__w, self._onMouse)
        return self._loop

    def __exit__(self, exc_type, exc_value, traceback):
        cv2.setMouseCallback(self.__w, lambda *_: None)

    def _onMouse(self, event, x, y, flags, param):
        if events[event] == 'EVENT_MOUSEMOVE':
            self.__xy = (x, y)
        elif events[event] == 'EVENT_LBUTTONDOWN':
            self.__pts.append((x, y))
        else:
            print('Ignoring:', events[event], x, y, flags, param)

    def _loop(self):
        while len(self.__pts) < self.__ct:
            numpy.copyto(self.__dst, self.__src)
            self.__an(self.__dst, self.__xy, self.__pts)
            x, y = self.__xy
            cv2.line(self.__dst, (x - self.__s, y), (x + self.__s, y), (0, 0, 255))
            cv2.line(self.__dst, (x, y - self.__s), (x, y + self.__s), (0, 0, 255))
            cv2.imshow(w, self.__dst)
            if cv2.waitKey(1) == 27:
                raise ValueError('ESC key')
        return self.__pts

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
    with MouseQuery(w, first, count=2, size=20, annot=a) as loop:
        box = loop()

    for frame in itertools.chain([first], m):
        a(frame, box[1], box)
        cv2.imshow(w, frame)
        cv2.waitKey(1) == 27 and exit(0)

# eof
