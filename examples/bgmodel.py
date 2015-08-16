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

def renewCap(c):
    c.capture.release()
    return c.duplicate()

if __name__ == '__main__':
    m = opencv.Capture.argtype(sys.argv[1] if sys.argv[1:] else 0)
    w = 'opencv'
    cv2.namedWindow(w, cv2.WINDOW_NORMAL)

    bgModelA = cv2.BackgroundSubtractorMOG()
    bgModelB = cv2.BackgroundSubtractorMOG()

    print('LEFT background model is generated on the fly')
    print('MIDDLE is the difference between background models')
    print('RIGHT background model is being generated from 200 frames of video')
    for (i, frame) in itertools.izip(range(200), m):
        bgModelB.apply(frame)
    else:
        print('done generating')

    diff = None
    for frame in renewCap(m):
        fgMaskA = bgModelA.apply(frame)
        fgMaskB = bgModelB.apply(frame)
        if diff is None:
            diff = numpy.empty_like(fgMaskA)
        cv2.absdiff(fgMaskA, fgMaskB, diff)

        cv2.imshow(w, numpy.hstack([fgMaskA, diff, fgMaskB]))
        cv2.waitKey(1) == 27 and exit(0)

# eof
