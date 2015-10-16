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

    first = next(iter(m))

    annot_p = lambda dst, pt: cv2.circle(dst, pt, 10, (255,) * 3)
    annot_bqr = lambda *a: \
            ( mouse.annotate_box(*a, color_fn=lambda *_: (0,255,255))
            , a[1] and mouse.annotate_quadrants(*a)
            , mouse.annotate_reticle(*a, color_fn=lambda *_: (0,0,255), size=25)
            )

    with mouse.MouseQuery(w, first.image, annot_fn=annot_bqr) as loop:
        point = loop()[0]
        annot_p(first.image, point)
    with mouse.MouseQuery(w, first.image, point_count=2, annot_fn=annot_bqr) as loop:
        box = loop()

    for fi in m:
        annot_p(fi.image, point)
        annot_bqr(fi.image, None, box[0], box)
        cv2.imshow(w, fi.image)
        cv2.waitKey(1) == 27 and exit(0)

# eof
