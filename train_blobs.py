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
import glob
import itertools
import argparse

import cv2
import numpy

import lib.iterutils as iterutils
import lib.cvutils as cvutils
import lib.train_blobs as train_blobs
import lib.blob_params as blob_params

def train_mainloop \
        ( sess
        , movies
        , expect_larva
        , path = None
        , window = None
        , verbose = False
        ):
    for mov in movies:
        c = cvutils.Capture.argtype(mov)
        train_blobs.train \
            ( sess
            , itertools.izip(c, itertools.repeat(expect_larva))
            , lambda expect, blobs: len(blobs) - expect
            , verbose = verbose
            , imshow_window = window
            )
        c.capture.release()
        if path:
            train_blobs.preserve_session(path, sess)
        if train_blobs.training_status(sess, '| after ...' + repr(mov[-20:]), verbose=True):
            print('Stopping conditions reached')
            return 0

def main(args):
    bucket_count = 2
    expect_larva = 10
    max_playthroughs = 999

    # make sure all movies are openable
    for mov in args.movies:
        c = cvutils.Capture.argtype(mov)
        c.capture.release()
    # infinite(ish) iterable over movie paths
    playlists = (blob_params.shuffled(args.movies) for _ in xrange(max_playthroughs))
    movies = itertools.chain.from_iterable(playlists)

    # create a window
    window = 'training examples' if args.window else None
    if args.window:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # training session
    # FIXME: strip extensions off args.session
    if args.session \
            and 1 == len(glob.glob(args.session + '*.p')) \
            and 1 == len(glob.glob(args.session + '*.npy')):
        sess = train_blobs.restore_session(args.session)
        train_blobs.print_session_info(sess)
        train_blobs.training_status(sess, '', verbose=True)
        print('Resuming..')
    else:
        sess = train_blobs.new_session \
            ( bucket_count
            , { 'minRepeatability':(long(0), long(10))
              , 'minThreshold':(0.0, 'maxThreshold')
              , 'maxThreshold':('minThreshold', 255.0)
              , 'thresholdStep':(5.0, 25.0)
              , 'minDistBetweenBlobs':(1.0, 400.0)
              , 'filterByArea':True
              , 'minArea':(1.0, 'maxArea')
              , 'maxArea':('minArea', 400.0)
              , 'filterByCircularity':True
              , 'minCircularity':(0.0, 'maxCircularity')
              , 'maxCircularity':('minCircularity', 1.0)
              , 'filterByColor':True
              , 'blobColor':(0, 255)
              , 'filterByConvexity':True
              , 'minConvexity':(0.0, 'maxConvexity')
              , 'maxConvexity':('minConvexity', 1.0)
              , 'filterByInertia':True
              , 'minInertiaRatio':(0.0, 'maxInertiaRatio')
              , 'maxInertiaRatio':('minInertiaRatio', 1.0)
              }
            )
        train_blobs.print_session_info(sess)
        if not args.session:
            print('WARNING: No data will be saved. Give -s/--session to save data.')
        if raw_input('Continue? [Y/n] ').lower().startswith('n'):
            return 0

    return train_mainloop \
        ( sess
        , movies
        , expect_larva
        , path = args.session
        , window = window
        , verbose = args.verbose
        )

if __name__ == '__main__':
    p = argparse.ArgumentParser()
#    p.add_argument \
#        ( 'count'
#        , metavar = 'C'
#        , type = int
#        , help = '''Integer count of larva expected in every frame of the given
#        movies.''')
    p.add_argument \
        ( 'movies'
        , metavar = 'M'
        , nargs = '+'
        , type = str
        , help = '''Movie file paths for training blob parameters.''')
    p.add_argument \
        ( '-s', '--session'
        , metavar = 'S'
        , type = str
        , help = '''File path prefix (will have extensions stripped and
        reapplied to it) of training state files. Data will be read from this
        location if it exists. New data will be saved to this location as
        training proceeds.''')
#   p.add_argument \
#       ( '-b', '--buckets'
#       , metavar = 'B'
#       , type = int
#       , help = '''How many buckets to squish the parameter space into (should
#       be at least 2).''')
    p.add_argument \
        ( '-w', '--window'
        , action = 'store_true'
        , help = '''Show the training examples in a window (slower).''')
    p.add_argument \
        ( '-v', '--verbose'
        , action = 'store_true'
        , help = '''Verbose execution.''')
    exit(main(p.parse_args()))


# eof

