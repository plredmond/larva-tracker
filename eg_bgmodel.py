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

import lib.cvutils as cvutils

if __name__ == '__main__':
    m = cvutils.Capture.argtype(sys.argv[1] if sys.argv[1:] else 0)
    w = 'opencv'
    cv2.namedWindow(w, cv2.WINDOW_NORMAL)

    bgModelA = cv2.BackgroundSubtractorMOG()
    bgModelB = cv2.BackgroundSubtractorMOG()

    print('LEFT background model is generated on the fly')
    print('MIDDLE is the difference between background models')
    print('RIGHT background model is being generated from 200 frames of video')
    for fi in itertools.islice(m, 200):
        bgModelB.apply(fi.image)
    else:
        print('done generating')

    diff = None
    for fi in m:
        fgMaskA = bgModelA.apply(fi.image)
        fgMaskB = bgModelB.apply(fi.image)
        if diff is None:
            diff = numpy.empty_like(fgMaskA)
        cv2.absdiff(fgMaskA, fgMaskB, diff)

        cv2.imshow(w, numpy.hstack([fgMaskA, diff, fgMaskB]))
        cv2.waitKey(1) == 27 and exit(0)

# eof
