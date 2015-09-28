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
import contextlib

import numpy
import cv2

import lib.track_blobs as trblobs
import lib.blob_params as blob_params
import lib.mouse as mouse
import lib.circles as circles
import lib.cviter as cviter
import lib.cvutils as cvutils

penny_diameter_mm = 19.05

def blobTracking(stream, debug=None):
    for (annotCur, annotHist, paths) in stream:
#       print('path count:', len(paths))
        yield [annotCur, annotHist]

def manual_crop(bounding_box, frames):
    '''(Int, Int, Int, Int), iter<ndarray> -> iter<ndarray>

       Crop frames to the given x, y, width, height.
    '''
    x0, y0, width, height = bounding_box
    x1 = x0 + width
    y1 = y0 + height
    return itertools.imap(lambda im: im[y0:y1, x0:x1, ...], frames)

def annot_bqr(*args):
    '''a MouseQuery annotator'''
    _, lmb, _, _ = args
    mouse.annotate_box(*args, color_fn=lambda *_: (0,255,255))
    if lmb:
        mouse.annotate_quadrants(*args)
    mouse.annotate_reticle(*args, color_fn=lambda *_: (0,0,255), size=25)

@contextlib.contextmanager
def window(name, *args, **kwargs):
    cv2.namedWindow(name, *args, **kwargs)
    yield
    # FIXME: doesn't actually close the window
    cv2.destroyWindow(name)

def petri_mask(petri_mean, frames, debug=None):
    petri_cx, petri_cy, petri_r = petri_mean
    ns = None
    for fr in frames:
        if ns is None:
            ns = \
                ( numpy.empty_like(fr)
                , numpy.zeros(fr.shape[:2] + (1,), fr.dtype)
                # TODO: do we need the extra dim?
                )
            cv2.circle(ns[1], (petri_cx, petri_cy), petri_r, 255, -1)
        out, mask = ns
        out.fill(0) # out[...] = 0
        cv2.bitwise_and(fr, fr, out, mask=mask)
        cviter._debugWindow(debug, petri_mask.func_name, ns)
        yield out

def main(args):

    # print args
    map(lambda s: print('{}: {}'.format(s, getattr(args, s))),
            filter(lambda s: s[0] != '_',
                sorted(dir(args))))
    print('= Frame shape:', next(args.movie.duplicate()).shape)

    # window (sink)
    windowName = '{0} - {1}'.format(path.basename(sys.argv[0]), args.movie.source)
    win = functools.partial(window, windowName, cv2.WINDOW_NORMAL)

    # movie (source)
    # TODO: push getting frame info into Capture
    # TODO: push forking n ways into Capture (since it allocates for every frame anyway)
    cue = lambda movie, step=1: itertools.islice \
            ( movie
            , args.drop
            , None if args.count is None else args.drop + args.count
            , step
            )
    get_movie = lambda **kwargs: cue(args.movie.duplicate(), **kwargs)

    # penny for scale
    # TODO: must print question on the frame somewhere
    with win():
        with mouse.MouseQuery \
                ( windowName
                , next(get_movie())
                , point_count = 2
                , annot_fn = annot_bqr
                ) as loop:
            penny_pts = loop()
    print('penny_pts', penny_pts)
    x0, x1 = sorted(max(x, 0) for x,_ in penny_pts)
    y0, y1 = sorted(max(y, 0) for _,y in penny_pts)
    penny_result = circles.find_circle \
            ( 15
            , 3.25
            , itertools.imap(lambda im: im[y0:y1, x0:x1, ...], get_movie(step=2))
            , blur = 8
            , param2 = 25
            , minFraction = 0.5
            , maxFraction = 1.5
            )
    if penny_result:
        penny_pct, penny_mean, penny_std = penny_result
        if penny_pct < 0.9:
            print("! Warning: The penny was detected in less than 90% of frames. Try re-running with `-d/--debug penny` to see what's failing.")
        print('= Mean penny diameter: {}px (standard deviation: {}px)'.format(penny_mean[2] * 2, penny_std[2]))
        mm_per_px = penny_diameter_mm / (penny_mean[2] * 2)
        print('= Scale: {} mm/px'.format(mm_per_px))
    else:
        print('= Penny wasn\'t found')
        exit(1)
    # TODO: include scale on analysis screen

    # petri dish for crop
    petri_result = circles.find_circle \
            ( 10
            , 15.0
            , get_movie(step=2)
            , blur = 10
            , param2 = 50
            , minFraction = 0.8
            , maxFraction = 1.2
            )
    if petri_result:
        petri_pct, petri_mean, petri_std = petri_result
        if petri_pct < 0.9:
            print("! Warning: The petri dish was detected in less than 90% of frames. Try re-running with `-d/--debug petri` to see what's failing.")
        print('= Mean petri dish diameter: {}px (standard deviation: {}px)'.format(petri_mean[2] * 2, petri_std[2]))
    else:
        print('= Petri dish wasn\'t found')
        exit(1)

    # crop
    petri_cx, petri_cy, petri_r = petri_mean
    petri_bbx = max(0, petri_cx - petri_r)
    petri_bby = max(0, petri_cy - petri_r)
    cropped = manual_crop \
        ( [petri_bbx, petri_bby, 2*petri_r, 2*petri_r]
        , petri_mask(petri_mean, get_movie())
        )

    # track
    params = { "filterByConvexity": False
             , "filterByCircularity": False
             , "filterByInertia": False
             , "filterByColor": False
             , "filterByArea": True
             , "minArea": 50.0
             , "maxArea": 250.0
             }
    disp = blobTracking(trblobs.trackBlobs \
        ( blob_params.mkDetector(params, verbose=True)
        , cropped
        #, debug = 'blobs'
        , anchor_match_dist=20
        , max_flow_err=20
        , blur_size=4
        ))

    cviter.displaySink(windowName, disp, ending=True)

sentinel = \
    {
    }

default = \
    { 'drop': 0
    }

if __name__ == '__main__':

    # test
    doctests = map(lambda m: (m, doctest.testmod(m)),
        [ None
        , cviter
        , trblobs
        , cvutils
        ])
    if any(test.failed for module, test in doctests):
        for module, test in doctests:
            print('{m}: {f} of {a} tests failed'.format \
                ( m = module.__name__ if module else __name__
                , f = test.failed
                , a = test.attempted
                ))
        exit(9)

    # args
    p = argparse.ArgumentParser()
    p.add_argument \
        ( 'movie'
        , type = cvutils.Capture.argtype
        , help = '''The path to the movie file to perform image tracking on.''')
    p.add_argument \
        ( '-d', '--drop'
        , default = default['drop']
        , metavar = 'D'
        , type = int
        , help = '''Number of frames to drop from the beginning of the analysis. (default {deft})'''.format(deft = default['drop']))
    p.add_argument \
        ( '-c', '--count'
        , metavar = 'D'
        , type = int
        , help = '''Number of frames to read from the movie file. (default: all)''')

    # main
    exit(main(p.parse_args()))

# eof
