from __future__ import \
    ( nested_scopes
    , generators
    , division
    , absolute_import
    , with_statement
    , print_function
    , unicode_literals
    )

import collections
import itertools

import numpy
import cv2

import lib.iterutils as iterutils
import lib.cviter as cviter

# feature set accessors
tlPointI = lambda itlArr, i: itlArr[i,:,:]
ilPointsAtT = lambda itlArr, t: itlArr[:,t,:]
ilPointsLatest = lambda itlArr: ilPointsAtT(itlArr, -1)
itlPointsLatest = lambda itlArr: ilPointsLatest(itlArr)[:,numpy.newaxis,:]

# annotation colors
Color = collections.namedtuple('Color', 'b g r a')
__activeFeature = Color(0, 0, 255, 255)
__inactiveFeature = Color(0, 0, 75, 255)
__activePath = Color(150, 100, 0, 255)
__inactivePath = Color(50, 50, 50, 255)

# tracking defaults
__maxCorners = 100
__qualityLevel = 0.25
__minDistance = 16

# annotation defaults
__circleRadius = int(round(__minDistance / 2))

TrackState = collections.namedtuple('TrackState', 'feats')

def trackCorners \
        ( frames
        , redetectInterval = None
        , maxCorners = __maxCorners
        , qualityLevel = __qualityLevel
        , minDistance = __minDistance
        , debug = None
        ):
    '''iter<ndarray>[, int][, int][, float][, int] -> iter<(ndarray, ndarray, bool, [ndarray<i,t,2>])>

       Iterate over frames of video.
       Yield the two frames tracking used as input, whether or not features were redetected on this iteration, and the accumulated history of corner feature sets and their optical-flow tracking updates.

       The yielded [ndarray<i,t,2>] contains each feature set until now.
       Each feature set is an ndarray with two coordinates for `i` points at `t` timesteps.
    '''
    # namespace for state
    ns = None
    # namespace mutators
    def setPointsLatest(itlArr):
        itlFeatArr = ns.feats.pop()
        ns.feats.append(numpy.hstack([itlFeatArr, itlArr]))
    # input stream
    stream = itertools.izip \
        ( cviter.buffering(2, frames)
        , itertools.chain([True], itertools.repeat(False)) \
          if redetectInterval is None else iterutils.trueEvery(redetectInterval)
        )
    # mainloop
    for ((im0, im1), redetectFeatures) in stream:
        # initialize ns if it hasn't already been
        if ns is None:
            ns = TrackState(feats=[])

        # verify datatypes
        # TODO: add assertion about im0 being gray?

        # make features
        if redetectFeatures or not ns.feats:
            ns.feats.append(cv2.goodFeaturesToTrack(
                im0, maxCorners, qualityLevel, minDistance))

        # update feature locations
        newPts, status, _ = cv2.calcOpticalFlowPyrLK(im0, im1, itlPointsLatest(ns.feats[-1]))
        setPointsLatest(newPts)
        # TODO: include status as extra coordinate-dimension

        cviter._debugWindow(debug, trackCorners.func_name, [im0, im1])
        yield (im0, im1, redetectFeatures, ns.feats)

def annotateFeatures \
        ( tracked
        , radius = __circleRadius
        , active = __activeFeature
        , inactive = __inactiveFeature
        , debug = None
        ):
    '''iter<...>[, int][, Color][, Color] -> iter<(ndarray, ndarray)>
    '''
    annot = None
    for (im0, _, redetected, featureHist) in tracked:
        # alloc
        if annot is None:
            annot = numpy.empty(im0.shape[:2] + (4,), im0.dtype)
        # proc
        annot[...] = 0
        for pt in ilPointsAtT(featureHist[-1], 0):
            cv2.circle(annot, tuple(pt), radius, active if redetected else inactive)
        # yield
        cviter._debugWindow(debug, annotateFeatures.func_name, [im0, annot])
        yield (im0, annot)

def annotatePaths \
        ( tracked
        , active = __activePath
        , inactive = __inactivePath
        , debug=None):
    '''iter<...>[, Color][, Color] -> iter<(ndarray, ndarray)>
    '''
    annot = None
    for (_, im1, _, featureHist) in tracked:
        # alloc
        if annot is None:
            annot = numpy.empty(im1.shape[:2] + (4,), im1.dtype)
        # proc
        annot[...] = 0
        curFI = len(featureHist) - 1
        for fI, features in enumerate(featureHist):
            pN, _, _ = features.shape
            for pI in range(pN):
                spans = iterutils.slidingWindow(2, tlPointI(features, pI))
                for (tI, (loc0, loc1)) in enumerate(spans):
                    cv2.line(annot, tuple(loc0), tuple(loc1),
                            active if fI == curFI else inactive)
        # yield
        cviter._debugWindow(debug, annotatePaths.func_name, [im1, annot])
        yield (im1, annot)

# eof
