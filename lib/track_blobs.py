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
import lib.blob_params as blob_params

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

#cv2.cvtColor(bim1, cv2.COLOR_GRAY2BGR, ns.annotCur)

TrackState = collections.namedtuple('TrackState', 'blobHist flowHist annotCur annotHist')

def trackBlobs(detector, frames, debug=None):
    '''iter<ndarray<x,y,3>>[, str] -> ...
    '''
    ns = None

    # fork
    framesA, framesB = cviter.fork(2, frames)

    # blob input can be anything
    blobInput = cviter.buffering(2, cviter.lift \
        ( lambda fr, denoise: cv2.blur(fr, (5, 5), denoise)
        , framesA
        ))

    # flow input should be 8-bit
    flowInput = cviter.buffering(2, cviter.gray(framesB))

    # loop
    for (bim0, bim1), (fim0, fim1) in itertools.izip(blobInput, flowInput):
        # allocate
        if ns is None:
            ns = TrackState \
                ( blobHist = [detector.detect(bim0)]
                , flowHist = []
                , annotCur = numpy.empty(bim1.shape[:2] + (3,), bim1.dtype)
                , annotHist = numpy.empty(bim1.shape[:2] + (3,), bim1.dtype)
                )
        ns.blobHist.append(detector.detect(bim1))
        bs0, bs1 = ns.blobHist[-2:]
        ps0 = numpy.array([b.pt for b in bs0], numpy.float32)
        ps1, status, err = cv2.calcOpticalFlowPyrLK(fim0, fim1, ps0)
        # TODO: store ps0..ps1 somewhere
        # TODO-LATER: match bs0 to bs1 based on updated location predictions in ps1

        # annotate current
        numpy.copyto(ns.annotCur, bim1)
        cv2.drawKeypoints(ns.annotCur, bs1, ns.annotCur, (50,150,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        for p0, p1 in zip(ps0, ps1):
            cv2.line(ns.annotCur, tuple(p0), tuple(p1), (0,0,255))

        # annotate history
        # TODO: loop to draw lines?
        numpy.copyto(ns.annotHist, bim1)
        for b in ns.blobHist:
            cv2.drawKeypoints(ns.annotHist, b, ns.annotHist, (50,150,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        if debug:
            bimdt = cv2.absdiff(bim0, bim1) # [...,None]
            fimdt = cv2.absdiff(fim0, fim1)[...,None]
            cviter._debugWindow(debug, trackBlobs.func_name, [bimdt, fimdt, ns.annotCur, ns.annotHist])
        yield [ns.annotCur, ns.annotHist]

# eof
