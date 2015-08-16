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
import itertools

import cv2
import numpy

import lib.opencv as opencv

if __name__ == '__main__':
    m = opencv.Capture.argtype(sys.argv[1] if sys.argv[1:] else 0)
    w = 'opencv'
    cv2.namedWindow(w, cv2.WINDOW_NORMAL)

    zip(m, range(5))
    ref = next(m)
    dst = numpy.empty_like(ref)
    opencv.explain('reference', ref)
    opencv.explain('destination', dst)
    for frame in m:
        cv2.absdiff(frame, ref, dst)

        cv2.imshow(w, numpy.hstack([frame, ref, dst]))
        cv2.waitKey(1) == 27 and exit(0)

# eof

