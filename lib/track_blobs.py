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

import cv2
import numpy

import lib.cviter as cviter

def printProps(obj):
    for attr in sorted(filter(lambda s: not s.startswith('__'), dir(obj))):
        print('\t{}: {}'.format(attr, getattr(obj, attr)))

def mkDetector(verbose=True, **kwargs):
    p = cv2.SimpleBlobDetector_Params()
    t = {a: type(getattr(p, a)) for a in dir(p) if not a.startswith('__')}
    for k, v in kwargs.items():
        assert k in t, '{} must be a property of {}'.format(repr(k), type(p).__name__)
        assert type(v) == t[k], '{} type must be {}, got a {}, {}'.format(k, t[k].__name__, type(v).__name__, v)
        setattr(p, k, v)
    verbose and printProps(p)
    return cv2.SimpleBlobDetector(p)

# supersillyus

# detect blobs in every frame
# use optical-flow between successive frames to assign new blobs to old blobs

# t0
# detect blobs and store

# t1
# detect blobs and store
# calculate optical flow of frame[t0], frame[t1], blobset[t0]
# assign blobset[t1] to blobset[t0]
#   - for status==1 flows

TrackState = collections.namedtuple('TrackState', 'blobHist flowHist annotCur annotHist')

def trackBlobs(frames, debug=None):
    '''iter<ndarray<x,y,3>>[, str] -> ...
    '''
    ns = None
    detect = lambda im: \
        ( permissive_detector.detect(im)
        , exclusive_detector.detect(im)
        )
    framesA, framesB = cviter.fork(2, frames)
    # blob input can be anything
    blobInput = cviter.buffering(2, cviter.lift \
        ( lambda fr, ns: cv2.blur(fr, (5, 5), ns)
        , framesA
        ))
    # flow input should be 8-bit
    flowInput = cviter.buffering(2, cviter.gray(framesB))
    for ((bim0, bim1), (fim0, fim1)) in itertools.izip(blobInput, flowInput):
        # allocate
        if ns is None:
            ns = TrackState \
                ( blobHist = [detect(bim0)]
                , flowHist = []
                , annotCur = numpy.empty(bim1.shape[:2] + (3,), bim1.dtype)
                , annotHist = numpy.empty(bim1.shape[:2] + (3,), bim1.dtype)
                )
        ns.blobHist.append(detect(bim1))
        ((bper0, bexc0), (bper1, bexc1)) = ns.blobHist[-2:]
#       print(b0[0])
#       p0 = None #numpy.array(, numpy.float32)
        import lib.cvutils as cvutils
#       cvutils.explain('input', p0)
#       p1, status, err = cv2.calcOpticalFlowPyrLK(t0, t1, p0)
#       cvutils.explain('output', p1)
        # match b0 to b1 based on updated location predictions in p1

        # annotate current
        #cv2.cvtColor(bim1, cv2.COLOR_GRAY2BGR, ns.annotCur)
        numpy.copyto(ns.annotCur, bim1)
        cv2.drawKeypoints(ns.annotCur, bper1, ns.annotCur, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.drawKeypoints(ns.annotCur, bexc1, ns.annotCur, (50,150,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # annotate history
        #cv2.cvtColor(bim1, cv2.COLOR_GRAY2BGR, ns.annotHist)
        numpy.copyto(ns.annotHist, bim1)
        for (bper, bexc) in ns.blobHist:
            cv2.drawKeypoints(ns.annotHist, bper, ns.annotHist, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.drawKeypoints(ns.annotHist, bexc, ns.annotHist, (50,150,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        if debug:
            bimdt = cv2.absdiff(bim0, bim1)[...,None]
            fimdt = cv2.absdiff(fim0, fim1) # [...,None]
            cviter._debugWindow(debug, trackBlobs.func_name, [bimdt, fimdt, ns.annotCur, ns.annotHist])
        yield [ns.annotCur, ns.annotHist]

# eof
