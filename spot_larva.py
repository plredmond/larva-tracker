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

import lib.opencv as opencv
import lib.iterutils as iterutils
import lib.funcutils as funcutils

def gray2color(src, dst=None):
    return cv2.merge([src, src, src], dst)

def explain(msg, im, pr=False):
    print(im.ndim, im.shape, im.dtype, im.size, msg)
    if pr:
        print(im)

# TODO: expose these defaulted args as options
def trackThings \
        ( imagePairIter
        , redetectInterval = None
        , maxCorners = 100
        , qualityLevel = 0.25
        , minDistance = 16
        ):

    # feature set accessors
    tlPointI = lambda itlArr, i: itlArr[i,:,:]
    ilPointsAtT = lambda itlArr, t: itlArr[:,t,:]
    ilPointsLatest = lambda itlArr: ilPointsAtT(itlArr, -1)
    itlPointsLatest = lambda itlArr: ilPointsLatest(itlArr)[:,numpy.newaxis,:]

    # colors
    activeFeature = (0, 0, 255)
    inactiveFeature = (0, 0, 75)
    activePath = (150, 100, 0)
    inactivePath = (50, 50, 50)

    # namespace for state
    TrackState = collections.namedtuple('TrackState', 'feats im0annot im1annot')
    ns = None

    # namespace mutators
    def setPointsLatest(itlArr):
        itlFeatArr = ns.feats.pop()
        ns.feats.append(numpy.hstack([itlFeatArr, itlArr]))

    # input stream
    stream = itertools.izip \
        ( imagePairIter
        , itertools.chain([True], itertools.repeat(False)) \
          if redetectInterval is None else \
          iterutils.trueEvery(redetectInterval)
        )

    # mainloop
    for ((im0, im1), redetectFeatures) in stream:
        # initialize ns if it hasn't already been
        if ns is None:
            ns = TrackState \
                ( feats=[]
                , im0annot=numpy.empty(im0.shape[:2] + (3,), im0.dtype)
                , im1annot=numpy.empty(im0.shape[:2] + (3,), im0.dtype)
                )
        # verify datatypes
        assert im0.shape == im1.shape and im0.dtype == im0.dtype == ns.im0annot.dtype

        # make features & local reference
        if redetectFeatures or not ns.feats:
            ns.feats.append(cv2.goodFeaturesToTrack(
                im0, maxCorners, qualityLevel, minDistance))
        latestFeatures = ns.feats[-1]

        # annotate found features
        gray2color(im0, ns.im0annot)
        for pt in ilPointsAtT(latestFeatures, 0):
            cv2.circle(ns.im0annot, tuple(pt), int(round(minDistance / 2)), activeFeature)

        newPts, status, _ = cv2.calcOpticalFlowPyrLK(im0, im1,
                itlPointsLatest(latestFeatures))
        setPointsLatest(newPts)
        # TODO: use status to stop annotating some points

        # annotate tracked movement
        gray2color(im1, ns.im1annot)
        curFI = len(ns.feats) - 1
        for fI, features in enumerate(ns.feats):
            pN, _, _ = features.shape
            for pI in range(pN):
                for (tI, (loc0, loc1)) in enumerate(iterutils.slidingWindow(2, tlPointI(features, pI))):
                    cv2.line(ns.im1annot, tuple(loc0), tuple(loc1),
                            activePath if fI == curFI else inactivePath)
            for point in ilPointsAtT(features, 0):
                cv2.circle(ns.im1annot, tuple(point), int(round(minDistance / 2)),
                        activeFeature if fI == curFI else inactiveFeature)
        yield ns

# - find the blob of moving larva in the center (area of interest)
#   - show area of interest as darkening mask in lhs
# - for analysis, ignore stuff outside of AOI
# - do feature detection on the hsl-tweaked AOI image
# - increase the diameter of the AOI according to the flow info
# - find the red penny, measure, erase it

def prepForTracking(triplets):
    ns = None
    for (t0, t1, t2) in triplets:
        # initialize ns if it hasn't already been
        if ns is None:
            ns = (numpy.empty_like(t0), numpy.empty_like(t0))
        # verify datatypes
        assert t0.shape == t1.shape == t2.shape == ns[0].shape \
                and t0.dtype == t1.dtype == t2.dtype == ns[0].dtype
        # process data
        cv2.absdiff(t0, t1, ns[0])
        cv2.absdiff(t1, t2, ns[1])
        yield ns

def sink(windowName, stream):
    for (frame, tracked) in stream:
        def showFn(n):
            cv2.imshow(windowName, numpy.hstack(
                ( frame
                , tracked.im1annot
                )))
            k = cv2.waitKey(n)
            k == 27 and exit(0)
            return k
        showFn(1)
    try:
        return showFn, tracked
    except UnboundLocalError:
        pass

def main(args):

    # print args
    map(lambda s: print('{}: {}'.format(s, getattr(args, s))),
            filter(lambda s: s[0] != '_',
                sorted(dir(args))))

    # create the window
    windowName = '{0} - {1}'.format(path.basename(sys.argv[0]), args.movie.source)
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

    # TODO: find a way to express fork/tee and join/zip with nesting structure, wrap it all up

    # main stream: drop frames and fork stream
    mA, mB = funcutils.pipeline \
        ( [ lambda m: itertools.islice(m, args.drop, None)
          , lambda m: itertools.tee(m, 2)
          ]
        , args.movie
        )

    # left side of fork: do image tracking
    tr = funcutils.pipeline \
        ( [ lambda m: itertools.imap(lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2GRAY, f), m)
          , lambda m: iterutils.slidingWindow(3, m)
          , lambda mmm: prepForTracking(mmm)
          , lambda dd: trackThings(dd, redetectInterval=args.redetect if args.redetect != -1 else None)
          ]
        , mA
        )

    # joined fork: display results
    r = funcutils.pipeline \
        ( [ lambda streams: itertools.izip(*streams)
          , lambda stream: sink(windowName, stream)
          ]
        , (mB, tr)
        )

    # analysis
    if r:
        show, result = r
        while True:
            show(0)
    else:
        print('No frames of video made it through to the analysis step.\nPerhaps you gave a `--drop` argument which was too high?')

sentinel = \
    { 'disable-redetect': -1
    }

default = \
    { 'redetect': 20
    , 'drop': 0
    }

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('movie',
            type=opencv.Capture.argtype,
            help='''The path to the movie file to perform image tracking on.''')
    p.add_argument('-r', '--redetect',
            default=default['redetect'], type=int, metavar='I',
            help='''Interval number of frames to wait before redoing feature
            selection. (default {deft} frames, use {dis} to disable
            redetection)'''.format(
                deft=default['redetect'], dis=sentinel['disable-redetect']))
    p.add_argument('-d', '--drop',
            default=default['drop'], type=int, metavar='D',
            help='''Number of frames to drop from the beginning of the analysis.
            (default {deft})'''.format(deft=default['drop']))
    main(p.parse_args())

# eof
