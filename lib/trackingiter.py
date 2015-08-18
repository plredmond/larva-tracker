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

# annotation colors
Color = collections.namedtuple('Color', 'b g r a')
__active = Color(0, 255, 0, 255)
__inactive = Color(255, 0, 0, 255)
__invalid = Color(0, 0, 255, 255)

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
    # input stream
    stream = itertools.izip \
        ( cviter.buffering(2, frames)
        , itertools.chain([True], itertools.repeat(False)) \
          if redetectInterval is None else iterutils.trueEvery(redetectInterval)
        )
    # mainloop
    for ((im0, im1), redetectFeatures) in stream:
        assert im0.ndim == 3 and im0.shape[2] == 1, 'single channel image'
        assert not (im0 == 0).all(), 'not blank'
        # initialize ns if it hasn't already been
        if ns is None:
            ns = TrackState(feats=[])

        # make features
        if redetectFeatures or not ns.feats:
            # detect points
            newPts = cv2.goodFeaturesToTrack(im0, maxCorners, qualityLevel, minDistance)
            assert newPts is not None, 'must produce features'
            # append them as the latest featureset
            # input: insert a sentinel status value of -1 to the 2nd position in axis 2 to indicate these are detected features
            ns.feats.append(numpy.insert(newPts, 2, -1, axis=2))

        # retrieve the latest feature set
        itpArr = ns.feats[-1]

        # find updated feature locations
        # input: fix on latest time, extract only x&y properties, restore empty time axis
        newPts, status, _ = cv2.calcOpticalFlowPyrLK(im0, im1, itpArr[:,-1,:2][:,numpy.newaxis,:])
        # concatenate new points on the time axis
        # input: insert status as additional property of each point
        ns.feats[-1] = numpy.concatenate([itpArr, numpy.insert(newPts, 2, status, axis=2)], axis=1)

        # insert new points to the latest position in the time axis (where `pts` input is the result of insert above)
        #_,T,_ = itpArr.shape
        #ns.feats[-1] = numpy.insert(itpArr, T, pts[:,0,:], axis=1)

        cviter._debugWindow(debug, trackCorners.func_name, [im0, im1])
        yield (im0, im1, redetectFeatures, ns.feats)

def annotateFeatures \
        ( pairs
        , radius = __circleRadius
        , active = __active
        , inactive = __inactive
        , debug = None
        ):
    '''iter<([ndarray<i,t,2>], ndarray)>[, int][, Color][, Color][, str] -> iter<ndarray>
    '''
    annot = None
    for ((_, _, redetected, featureHist), im) in pairs:
        if annot is None:
            annot = numpy.empty(im.shape[:2] + (3,), im.dtype)
        annot[...] = im[...]
        itpArr = featureHist[-1]
        for (x, y, s) in itpArr[:,0,:]:
            assert s == -1, 'features are found-points (status is -1, not 0 or 1)'
            cv2.circle(annot, (x, y), radius, active if redetected else inactive)
        cviter._debugWindow(debug, annotateFeatures.func_name, [im, annot])
        yield annot

def annotatePaths \
        ( pairs
        , active = __active
        , inactive = __inactive
        , invalid = __invalid
        , debug=None):
    '''iter<([ndarray<i,t,2>], ndarray)>[, Color][, Color][, str] -> iter<ndarray>
    '''
    annot = None
    for ((_, _, _, featureHist), im) in pairs:
        if annot is None:
            annot = numpy.empty(im.shape[:2] + (3,), im.dtype)
        annot[...] = im[...]
        cur_f = len(featureHist) - 1
        for f, itpArr in enumerate(featureHist):
            I, _, _ = itpArr.shape
            for i in range(I):
                for (x0, y0, s0), (x1, y1, s1) in iterutils.slidingWindow(2, itpArr[i,:,:]):
                    cv2.line(annot, (x0, y0), (x1, y1),
                            (active if f == cur_f else inactive)
                            if s0 and s1 else invalid)
        cviter._debugWindow(debug, annotatePaths.func_name, [im, annot])
        yield annot

# eof
