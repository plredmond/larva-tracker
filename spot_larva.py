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
import doctest

import numpy
import cv2

import lib.cviter as cviter
import lib.trackingiter as triter
import lib.opencv as opencv
import lib.iterutils as iterutils
import lib.funcutils as funcutils

def cornerTracking(stream, debug=None):
    ns = None
    for ((_, _, redetect, featureHist), featAnnot, pathAnnot) in stream:
        if ns is None:
            pass

        if redetect and len(featureHist) > 1:
            itpArr0, itpArr1 = featureHist[-2:]
            I0,T,_ = itpArr0.shape
            I1,_,_ = itpArr1.shape
            print('Merge {} and {} points and reduce to 10'.format(I0, I1))
            print(itpArr0[:,T-1,:].shape)
            print(itpArr1[:,0,:].shape)

        elif redetect:
            itpArr = featureHist[-1]
            I,_,_ = itpArr.shape
            print('Reduce {} points to 10'.format(I))
            print(itpArr[:,0,:].shape)

        cviter._debugWindow(debug, cornerTracking.func_name, [pathAnnot, featAnnot])
        yield (pathAnnot, featAnnot)

def main(args):
    # print args
    map(lambda s: print('{}: {}'.format(s, getattr(args, s))),
            filter(lambda s: s[0] != '_',
                sorted(dir(args))))
    windowName = '{0} - {1}'.format(path.basename(sys.argv[0]), args.movie.source)
    cue = lambda m: itertools.islice(m, args.drop, None)

    # TODO: find the red penny, measure, erase it

    ###
    # source the video frames from the raw footage or from the movement mask
    if args.cam:
        movie = opencv.Capture.argtype(0)
    elif args.mask:
        moves = cviter.movementMask(cue(args.movie.duplicate()))
        movie = cviter.lift \
            ( lambda fr, ns: \
                ( ns.fill(0)
                , cv2.bitwise_and(fr, fr, ns, mask=moves.roi_fn(moves.mask))
                )
            , itertools.imap(moves.roi_fn, cue(args.movie))
            )
    else:
        movie = cue(args.movie)

    movieA, movieB = itertools.tee(movie, 2)

    ###
    # prep the frames for tracking
    if args.prep == 'gray':
        trinputA = cviter.gray(movieA)
    elif args.prep == 'motion':
        trinputA = cviter.motion(cviter.gray(movieA))
    elif args.prep == 'bg':
        trinputA = itertools.dropwhile(lambda fr: (fr == 0).all(),
                cviter.fgMask(cviter.gray(movieA)))
    elif args.prep == 'blur':
        trinputA = cviter.lift \
            ( lambda dt, ns: \
                ( cv2.GaussianBlur(dt, (3, 3), 0, ns)
                , cv2.threshold(ns, 190, 255, cv2.THRESH_BINARY, ns)
                )
            , cviter.gray(movieA)
            )

    ###
    # track in any way possible
    if args.tracking == 'corner':
        # TODO: try different args to goodFeaturesToTrack and calcOpticalFlowPyrLK
        trackAA = triter.trackCorners(trinputA, redetectInterval=args.redetect if args.redetect != -1 else None)
        trackAAA, trackAAB, trackAAC = itertools.tee(trackAA, 3)
        featsAAA = triter.annotateFeatures(itertools.imap(lambda tr: (tr, tr[0]), trackAAA))
        pathsAAB = triter.annotatePaths(itertools.izip(trackAAB, movieB))
        disp = cornerTracking(itertools.izip(trackAAC, featsAAA, pathsAAB))
    elif args.tracking == 'contour':
        raise NotImplementedError()
    elif args.tracking == 'blob':
        params = cv2.SimpleBlobDetector_Params()
        print(dir(params))
        params.minThreshold = 10
        params.maxThreshold = 200
        #params.filterByArea = True
        #params.minArea = 1500
        #params.filterByCircularity = True
        #params.minCircularity = 0.1
        #params.filterByConvexity = True
        #params.minConvexity = 0.87
        #params.filterByInertia = True
        #params.minInertiaRatio = 0.01

        detector = cv2.SimpleBlobDetector(params)
        def trackBlobs(fr, ns):
            keypoints = detector.detect(fr)
            print('keypoints:', len(keypoints))
            assert keypoints
            cv2.drawKeypoints(fr, keypoints, ns, (0,0,255),
                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        disp = cviter.lift(trackBlobs, trinputA,
                shapefn = lambda fr: fr.shape[:2] + (3,))

    cviter.displaySink(windowName, disp, ending=True)

sentinel = \
    { 'disable-redetect': -1
    , 'preparation-choices': {'gray', 'motion', 'bg', 'blur'}
    , 'tracking-choices': {'corner', 'blob', 'contour'}
    }

default = \
    { 'redetect': 20
    , 'drop': 0
    , 'prep': 'blur'
    , 'tracking': 'corner'
    }

if __name__ == '__main__':
    doctests = map(lambda m: (m, doctest.testmod(m)),
        [ None
        , cviter
        , triter
        , opencv
        , iterutils
        , funcutils
        ])
    if any(test.failed for module, test in doctests):
        for module, test in doctests:
            print('{m}: {f} of {a} tests failed'.format \
                ( m = module.__name__ if module else __name__
                , f = test.failed
                , a = test.attempted
                ))
        exit(9)

    p = argparse.ArgumentParser()

    # movie file
    p.add_argument \
        ( 'movie'
        , type = opencv.Capture.argtype
        , help = '''The path to the movie file to perform image tracking on.''')
    p.add_argument \
        ( '-d', '--drop'
        , default = default['drop']
        , metavar = 'D'
        , type = int
        , help = '''Number of frames to drop from the beginning of the analysis. (default {deft})'''.format(deft = default['drop']))
    # OR cam
    p.add_argument \
        ( '-c', '--cam'
        , action = 'store_true'
        , help = '''If given, ignore the file argument and the unmask argument. Use the computer's webcam instead.''')

    # preproccessing
    p.add_argument \
        ( '-p', '--prep'
        , choices = sentinel['preparation-choices']
        , default = default['prep']
        , metavar = 'P'
        , help = '''Type of image segmentation preparation to perform on images before tracking.
            "gray" turns them gray;
            "blur" blurs and thresholds gray images;
            "bg" uses a foreground/background model to isolate larva;
            "motion" finds the absolute difference between successive gray images;
            (default {deft})'''.format(deft=default['prep']))
    p.add_argument \
        ( '-u', '--unmask'
        , action = 'store_false'
        , dest = 'mask'
        , help = '''If given, do not produce or utilize the movement mask. The default is to use this mask.''')

    # corner tracking
    p.add_argument \
        ( '-r', '--redetect'
        , default = default['redetect']
        , metavar = 'I'
        , type = int
        , help = '''Interval number of frames to wait before redoing feature selection. (default {deft} frames, use {dis} to disable redetection)'''.format \
            ( deft = default['redetect']
            , dis = sentinel['disable-redetect']))
    # OR other kinds of tracking
    p.add_argument \
        ( '-t', '--tracking'
        , choices = sentinel['tracking-choices']
        , default = default['tracking']
        , metavar = 'T'
        , help = '''Type of tracking to perform on images.
            "blob" tracks elongated semiconcave blobs;
            "contour" tracks brightness and color contours filtered by size;
            "corner" tracks corner features with optical-flow;
            (default {deft})'''.format(deft=default['tracking']))

    main(p.parse_args())

# eof
