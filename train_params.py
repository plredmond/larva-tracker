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
import pickle
import time

import cv2
import numpy

import lib.cvutils as cvutils
import lib.blob_params as blob_params
import lib.track_blobs as track_blobs

if __name__ == '__main__':
    trainfile = 'paramtrain-{}'.format(int(time.time()))
    show = False

    # make sure all are openable
    for path in sys.argv[1:]:
        c = cvutils.Capture.argtype(path)
        c.capture.release()

    # make a window
    w = 'opencv'
    cv2.namedWindow(w, cv2.WINDOW_NORMAL)

    # training space
    param_space = \
        { 'minRepeatability':(long(0), long(10))
        , 'minThreshold':(0.0, 'maxThreshold')
        , 'maxThreshold':('minThreshold', 255.0)
        , 'thresholdStep':(5.0, 25.0)
        , 'minDistBetweenBlobs':(1.0, 400.0)
        , 'filterByArea':True
        , 'minArea':2.0#(1.0, 'maxArea')
        , 'maxArea':400.0#('minArea', 400.0)
        , 'filterByCircularity':False
        , 'minCircularity':0.0#(0.0, 'maxCircularity')
        , 'maxCircularity':0.0#('minCircularity', 1.0)
        , 'filterByColor':False
        , 'blobColor':0#(0, 255)
        , 'filterByConvexity':False
        , 'minConvexity':0.0#(0.0, 'maxConvexity')
        , 'maxConvexity':0.0#('minConvexity', 1.0)
        , 'filterByInertia':False
        , 'minInertiaRatio':0.0#(0.0, 'maxInertiaRatio')
        , 'maxInertiaRatio':0.0#('minInertiaRatio', 1.0)
        }

    # train
    bucketCt = 4
    bucketd = blob_params.paramBuckets(bucketCt, blob_params.paramRanges(param_space))
    index, protoshape = blob_params.trainingState(bucketCt, param_space)
    state = numpy.zeros(protoshape + [3])
    counts = state[..., 0]
    errsums = state[..., 1]
    uerrsums = state[..., 2]

    raw_input('Continue? [Y/n] ').lower().startswith('n') and exit(0)

    # mainloop
    for path in blob_params.shuffled(sys.argv[1:]):
        print('Starting', repr(path))
        c = cvutils.Capture.argtype(path)
        for i, fr in enumerate(c):
            print('frame', i)

            paramd = blob_params.randParams(param_space)
            print('\n'.join('  {}: {}'.format(k, paramd[k]) for k in index))
            blobs = track_blobs.mkDetector(verbose=not i, **paramd).detect(fr)
            signed_err = len(blobs) - 10 # every video has 10 larva
            print('-> signed_err', signed_err)

            path = tuple(bucketd[k](paramd[k]) for k in index)
            print('-> bucket path', path)
            counts[path] += 1
            errsums[path] += signed_err
            uerrsums[path] += abs(signed_err)

            if show:
                cv2.drawKeypoints(fr, blobs, fr, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.imshow(w, fr)
                cv2.waitKey(1) == 27 and exit(0)

        # preserve state
        with open('{}.npy'.format(trainfile), 'wb') as npyfile:
            numpy.save(npyfile, state)
            with open('{}.p'.format(trainfile), 'wb') as pfile:
                pickle.dump( \
                    { 'param_space': param_space
                    , 'bucket_count':  bucketCt
                    , 'index': index
                    , 'state-filename': npyfile.name
                    }, pfile)

        # release resources
        c.capture.release()

        # check stopping conditions
        print(counts.min(), counts.max())
        # what % of counts are greater than they were when we started this video ?
        # what % of counts are 10+ they were when we started this video ?
        # do we have enough information to calculate gradient for each feature and adjust the rand window?
        # visualization of avg errors?

# eof

