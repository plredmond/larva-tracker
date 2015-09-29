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

import cv2
import numpy

import lib.cviter as cviter

def find_circle(min_ct, max_std, *args, **kwargs):
    '''int, float, <args of circles_iter> -> float, ndarray<1,3> ndarray<1,3>

       Search each frame of a video for a singleton circle.

       Terminate search early if at least `min_ct` frames contained a singleton
       circle and the standard deviation of their location and radius is less
       than `max_std`.

       Maybe return (else `None` if the circle was never detected)
        ( float # fraction of searched frames in which one circle was detected
        , numpy.array([x, y, radius]) # mean properties over all singleton circles
        , numpy.array([x, y, radius]) # standard deviation of properties over all singleton circles
        )
    '''
    print('> Finding circle location & radius..')
    arrs = []
    for i, result in enumerate(circles_iter(*args, **kwargs)):
        ct, _ = result.shape
        if ct == 1:
            arrs.append(result)
        print('> Frame %d' % i, end='\r')
        sys.stdout.flush()
        # check early termination condition
        if len(arrs) > min_ct and (numpy.concatenate(arrs, axis=0).std(axis=0) < max_std).all():
            print('> Found after just %d frames' % i)
            break
    if arrs:
        arr = numpy.concatenate(arrs, axis=0)
        return len(arrs) / i, arr.mean(axis=0), arr.std(axis=0)

def circles_iter \
        ( frames
        , blur=None
        , param1 = None
        , param2 = None
        , minFraction = None
        , maxFraction = None
        , debug = None
        ):
    '''iter<ndarray<y,x,bgr>>[, int][, int][, int][, float][, float][, str] -> iter<...>

       Locate a circle in each frame.
        If given, blur with a square kernel `blur` pixels to a side.
        If given, apply param1 and param2 to cv2.HoughCircles.
        If given, return circles with diameter between minFraction and maxFraction of the small side of the frame.
    '''
    ns = None
    for fr in frames:
        if ns is None:
            ns = \
                ( numpy.empty(fr.shape[:2] + (1,), fr.dtype)
                , numpy.empty_like(fr)
                )
        det, debug_im = ns
        # convert to gray; optionally blur
        cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY, det)
        cv2.equalizeHist(det, det)
        if blur is not None:
            cv2.blur(det, (blur, blur), det)
        # find circles
        size = min(fr.shape[:2])

        params = {}
        if param1 is not None:
            params['param1'] = param1
        if param2 is not None:
            params['param2'] = param2
        if minFraction is not None:
            params['minRadius'] = int(size * minFraction / 2)
        if maxFraction is not None:
            params['maxRadius'] = int(size * maxFraction / 2)

        ret = cv2.HoughCircles(det, cv2.cv.CV_HOUGH_GRADIENT, 1, minDist=size, **params)
        circles = numpy.empty((0, 3)) if ret is None else ret[0, ...]
        if debug:
            numpy.copyto(debug_im, fr)
            [annot_target(c[0], c[1], c[2], debug_im) for c in circles]
            cviter._debugWindow(debug, circles_iter.func_name, ns)
        yield circles

def annot_target(x, y, r, im):
    pt = x, y
    cv2.circle(im, pt, r, (0,255,0), 2)
    cv2.circle(im, pt, 2, (0,0,255), 3)
    return im

# eof
