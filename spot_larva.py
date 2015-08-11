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
import collections
import os.path as path
import functools
import itertools
import argparse

import numpy
import cv2

class Capture(collections.namedtuple('Capture', 'source capture')):
    __slots__ = ()
    @classmethod
    def argtype(cls, source):
        '''str/int -> Capture
           Construct a Capture from the path/device or give an argument type error.
        '''
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            return Capture(source, cap)
        else:
            raise argparse.ArgumentTypeError('source {0}'.format(source))
    def __iter__(self):
        return self
    def next(self):
        '''-> numpy.ndarray'''
        ret, frame = self.capture.read()
        assert (ret and frame is not None) or (not ret and frame is None), 'ret and frame must agree'
        if ret:
            return frame
        else:
            self.capture.release()
            raise StopIteration

def slidingWindow(size, iterable):
    '''int iterable<a> -> iterable<[a]>
       Iterate over every `size` length subsequence of `iterable`.
    '''
    buff = collections.deque([], size)
    for v in iterable:
        buff.append(v)
        if len(buff) == size:
            yield list(buff)

def trueEvery(n):
    '''n -> iterable<bool>'''
    ct = 0
    while True:
        yield ct == 0
        ct = (ct + 1) % n

def flip(f0):
    '''(a, b, ... -> c) -> (..., b, a -> c)'''
    @functools.wraps(f0)
    def f1(*args, **kwargs):
        return f0(*args[::-1], **kwargs)
    return f1

def curry(f0):
    '''([a, b, ...] -> c) -> (a, b, ... -> c)'''
    @functools.wraps(f0)
    def f1(*args, **kwargs):
        return f0(args, **kwargs)
    return f1

def uncurry(f0):
    '''(a, b, ... -> c) -> ([a, b, ...] -> c)'''
    @functools.wraps(f0)
    def f1(args, **kwargs):
        return f0(*args, **kwargs)

PastFuture = collections.namedtuple('PastFuture', 'past future')
preprocess = \
    [
#     lambda arrs: arrs.past
      lambda arrs: cv2.absdiff(arrs.future, arrs.past)
    , lambda arr: cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    , lambda arr: cv2.merge([arr, arr, arr])
    ]

ap = lambda datum, fun: fun(datum)
pipeline = functools.partial(functools.reduce, ap)

# apply :: (a, b, ... -> c), [a, b, ...] -> c
# uncurry the first arg to change its args to a seq
#       :: ([a, b, ...] -> c), [a, b, ...] -> c
# simplify all occurances of [a, b, ...] to d (removing all instances of a and b)
#       :: (d -> c), d -> c
# flip the argument order
# ap    :: a, (a -> b) -> b


# - find the blob of moving larva in the center (area of interest)
#   - show area of interest as darkening mask in lhs
# - for analysis, ignore stuff outside of AOI
# - do feature detection on the hsl-tweaked AOI image
# - increase the diameter of the AOI according to the flow info
# - find the red penny, measure, erase it

def sink(window, accum, cur):
    '''str -> ? -> (bool, [numpy.ndarray]) -> ?'''
    #features, tracks = accum
    redetectFeatures, (past, future) = cur

    prepped = pipeline(preprocess, PastFuture(past, future))

    if redetectFeatures:
        #features = goodFeaturestoTrack
        #features = goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance[, corners[, mask[, blockSize[, useHarrisDetector[, k]]]]]) -> corners
        pass

    cv2.imshow(window, numpy.hstack((past, prepped)))
    cv2.waitKey(1) == 27 and exit(0)
    return None

def main(args):
    print(repr(args))
    windowName = '{0} - {1}'.format(path.basename(sys.argv[0]), args.movie.source)
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    result = functools.reduce \
        ( functools.partial(sink, windowName)
        , itertools.izip
            ( trueEvery(10)
            , slidingWindow(2, args.movie)
            )
        , None
        )

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('movie', type=Capture.argtype)
    main(p.parse_args())

# eof
