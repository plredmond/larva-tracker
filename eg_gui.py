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
    movie = itertools.chain([first], m)

    annot_p = lambda dst, pt: cv2.circle(dst, pt, 10, (255,) * 3)
    annot_bqr = lambda *a: \
            ( mouse.annotate_box(*a, color_fn=lambda *_: (0,255,255))
            , a[1] and mouse.annotate_quadrants(*a)
            , mouse.annotate_reticle(*a, color_fn=lambda *_: (0,0,255), size=25)
            )

    with mouse.MouseQuery(w, first, annot_fn=annot_bqr) as loop:
        point = loop()[0]
        annot_p(first, point)
    with mouse.MouseQuery(w, first, point_count=2, annot_fn=annot_bqr) as loop:
        box = loop()

    for frame in movie:
        annot_p(frame, point)
        annot_bqr(frame, None, box[0], box)
        cv2.imshow(w, frame)
        cv2.waitKey(1) == 27 and exit(0)

# eof
