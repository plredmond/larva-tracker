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

import lib.mouse as mouse
import lib.cvutils as cvutils

if __name__ == '__main__':
    m = cvutils.Capture.argtype(sys.argv[1] if sys.argv[1:] else 0)
    w = 'opencv'
    cv2.namedWindow(w, cv2.WINDOW_NORMAL)

    first = next(m)
    def annot_point(dst, pt):
        cv2.circle(dst, pt, 10, (25,125,0))
    def annot_box(dst, xy, pts):
        c = time.time() % 1 * 255
        assert 0 <= c < 255
        if pts:
            cv2.rectangle(dst, pts[0], xy, [c] * 3)
    with mouse.MouseQuery(w, first, big_reticle_color=(104,0,255)) as loop:
        point = loop()[0]
        annot_point(first, point)
    with mouse.MouseQuery(w, first, 2, 20, annot_box, (0,0,255), (0,255,255)) as loop:
        box = loop()

    for frame in itertools.chain([first], m):
        annot_point(frame, point)
        annot_box(frame, box[1], box)
        cv2.imshow(w, frame)
        cv2.waitKey(1) == 27 and exit(0)

# eof
