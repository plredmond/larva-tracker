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
    cv2.namedWindow(w, flags=cv2.WINDOW_NORMAL)

    ref = next(itertools.islice(iter(m), None))
    dst = numpy.empty_like(ref.image)
    cvutils.explain('reference', ref.image)
    cvutils.explain('destination', dst)
    for fi in m:
        cv2.absdiff(fi.image, ref.image, dst)

        cv2.imshow(w, numpy.hstack([fi.image, ref.image, dst]))
        cv2.waitKey(1) == 27 and exit(0)

# eof

