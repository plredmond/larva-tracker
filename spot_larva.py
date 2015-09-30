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

def blobTracking(stream, debug=None):
    for (annotCur, annotHist, paths) in stream:
        yield [annotCur, annotHist]

def manual_crop(bounding_box, frames):
    '''(Int, Int, Int, Int), iter<ndarray> -> iter<ndarray>

       Crop frames to the given x, y, width, height.
    '''
    x0, y0, width, height = bounding_box
    x1 = x0 + width
    y1 = y0 + height
    return itertools.imap(lambda im: im[y0:y1, x0:x1, ...], frames)

@contextlib.contextmanager
def window(name, *args, **kwargs):
    cv2.namedWindow(name, *args, **kwargs)
    yield
    # FIXME: doesn't actually close the window
    cv2.destroyWindow(name)

def annot_bqr(*args):
    '''a MouseQuery annotator which facilitates the selection of boxes'''
    _, lmb, _, _ = args
    mouse.annotate_box(*args, color_fn=lambda *_: (0,255,255))
    if lmb:
        mouse.annotate_quadrants(*args)
    mouse.annotate_reticle(*args, color_fn=lambda *_: (0,0,255), size=25)

def coin_for_scale(window_name, coin_diameter_mm, frameinfos, debug=None):
    # fetch the first frame of video & info
    first = next(frameinfos)
    frameinfos = itertools.chain([first], frameinfos)
    w, h = first.image.shape[:2]
    # query for a general area on the frame
    with window(window_name, cv2.WINDOW_NORMAL):
        with mouse.MouseQuery \
                ( window_name
                , first.image
                , point_count = 2
                , annot_fn = annot_bqr
                ) as loop:
            pts = loop()
    x0, x1 = sorted(max(x, 0) for x,_ in pts)
    y0, y1 = sorted(max(y, 0) for _,y in pts)
    # find a circle on that spot of frame
    result = circles.find_circle \
            ( 15
            , 3.25
            , itertools.imap(lambda fi: fi.image[y0:y1, x0:x1, ...], frameinfos)
            , blur = 8
            , param2 = 25
            , minFraction = 0.5
            , maxFraction = 1.5
            , debug = debug == 'coin' and debug
            )
    if result is None:
        print('! Error: Coin was not detected.')
        print('!    Try re-running with `-d/--debug coin` to see what is failing.')
        return
    pct, mean, std = result
    if pct < 0.9:
        print('! Warning: Coin was detected in less than 90% of frames.')
        print('!    Try re-running with `-d/--debug coin` to see what is failing.')
    # TODO: add a warning case for high standard deviation
    # TODO: combine this with the very similar paragraph in petri_for_crop
    # print interesting results
    uc, ud = mean[:2], mean[2] * 2
    sc, sr = std[:2], std[2]
    print('= Mean coin diameter: {}px (standard deviation: {}px)'.format(ud, sr))
    mm_per_px = coin_diameter_mm / ud
    print('= Scale: {} mm/px'.format(mm_per_px))
    return mm_per_px, [x0, y0, 0] + mean, std

def petri_for_crop(frameinfos, debug=None):
    result = circles.find_circle \
            ( 10
            , 15.0
            , itertools.imap(lambda fi: fi.image, frameinfos)
            , blur = 10
            , param2 = 50
            , minFraction = 0.8
            , maxFraction = 1.2
            , debug = debug == 'petri' and debug
            )
    if result is None:
        print('! Error: Petri dish was not detected.')
        print('!    Try re-running with `-d/--debug petri` to see what is failing.')
        return
    pct, mean, std = result
    if pct < 0.9:
        print('! Warning: Petri dish was detected in less than 90% of frames.')
        print('!    Try re-running with `-d/--debug petri` to see what is failing.')
    # TODO: add a warning case for high standard deviation
    # TODO: combine this with the very similar paragraph in coin_for_scale
    # print interesting results
    uc, ud = mean[:2], mean[2] * 2
    sc, sr = std[:2], std[2]
    print('= Mean petri dish diameter: {}px (standard deviation: {}px)'.format(ud, sr))
    return mean, std

def mask_circle(circle_x_y_r, frames, debug=None):
    c = tuple(circle_x_y_r[:2])
    r = circle_x_y_r[2]
    ns = None
    for fr in frames:
        if ns is None:
            ns = \
                ( numpy.empty_like(fr)
                , numpy.zeros(fr.shape[:2], fr.dtype)
                )
            cv2.circle(ns[1], c, r, 255, -1)
        out, mask = ns
        out.fill(0)
        cv2.bitwise_and(fr, fr, out, mask=mask)
        cviter._debugWindow(debug, mask_circle.func_name, ns)
        yield out

def circle_bb(circle_x_y_r, frame_w_h):
    '''Extract clamped and unclamped bounding boxes for the circle.
       Convert circle coordinates to clamped space.

       Return
        ( ([x, y], [w, h]) # unclamped-bounding-box in frame-space
        , ([x, y], [w, h]) # clamped-bounding-box in frame-space
        , [x, y] # unclamped-circle-point in clamped-bounding-box-space
        )

       >>> from numpy import array
       >>> flatten = lambda ((xy, wh), (cxy, cwh), cc): [xy, wh, cxy, cwh, cc]
       >>> xs = lambda r: map(lambda (x,_): x, flatten(r))
       >>> ys = lambda r: map(lambda (_,y): y, flatten(r))
       >>> # test the xs
       >>> xs(circle_bb([1, 0, 2], [4, 0]))
       [-1, 4, 0, 3, 1]
       >>> xs(circle_bb([2, 0, 2], [4, 0]))
       [0, 4, 0, 4, 2]
       >>> xs(circle_bb([3, 0, 2], [4, 0]))
       [1, 4, 1, 3, 2]
       >>> xs(circle_bb([2, 0, 1], [4, 0]))
       [1, 2, 1, 2, 1]
       >>> # test the ys
       >>> ys(circle_bb([0, 1, 2], [0, 4]))
       [-1, 4, 0, 3, 1]
       >>> ys(circle_bb([0, 2, 2], [0, 4]))
       [0, 4, 0, 4, 2]
       >>> ys(circle_bb([0, 3, 2], [0, 4]))
       [1, 4, 1, 3, 2]
       >>> ys(circle_bb([0, 2, 1], [0, 4]))
       [1, 2, 1, 2, 1]
    '''
    clamp = lambda lb, n, ub: numpy.maximum(lb, numpy.minimum(n, ub))
    c = numpy.array(circle_x_y_r[:2])
    r = circle_x_y_r[2]
    fd = numpy.array(frame_w_h)
    # unclamped-bounding-box in frame-space
    bb0 = c - r
    bb1 = c + r
    bbd = bb1 - bb0
    # clamped-bounding-box in frame-space
    cl = lambda n: clamp(0, n, fd)
    cbb0 = cl(bb0)
    cbb1 = cl(bb1)
    cbbd = cbb1 - cbb0
    # unclamped-circle-point in clamped-bounding-box-space
    cc = numpy.where(bb0 > 0, r, r + bb0)
    return \
        ( (bb0, bbd)
        , (cbb0, cbbd)
        , cc
        )

def main(args):

    # print args
    map(lambda s: print('{}: {}'.format(s, getattr(args, s))),
            filter(lambda s: s[0] != '_',
                sorted(dir(args))))
    print('= Frame width x height:', args.movie.frame_width, 'x', args.movie.frame_height)

    # window (sink)
    windowName = '{0} - {1}'.format(path.basename(sys.argv[0]), args.movie.source)
    win = functools.partial(window, windowName, cv2.WINDOW_NORMAL)

    # movie (source)
    cue = lambda step=None: args.movie[args.beginning:args.ending:step]

    # coin for scale
    # TODO: must print question on the frame somewhere
    # TODO: compose the penny into the top left of the frames before they are displayed (in blobTracking)
    mm_per_px, ucoin, scoin = coin_for_scale(windowName, args.coin_diameter, cue(step=3), debug=args.debug)

    # petri dish for crop
    upetri, spetri = petri_for_crop(cue(step=3), debug=args.debug)
    _, ((cbbx, cbby), (cbbw, cbbh)), cc = circle_bb(upetri, (args.movie.frame_width, args.movie.frame_height))
    cupetri = numpy.concatenate((cc, upetri[2:]))

    cropped = cviter.applyTo \
            ( lambda fi: fi.image
            , lambda fi, im: fi._replace(image=im)
            , lambda upstream: \
                    manual_crop([cbbx, cbby, cbbw, cbbh],
                        mask_circle(upetri, upstream))
            , cue()
            )

    # track
    cupetri_half = numpy.concatenate((cupetri[:2], 0.5 * cupetri[2:]))
    flagger = trblobs.gen_flagger(cupetri_half)
    params = { "filterByConvexity": False
             , "filterByCircularity": False
             , "filterByInertia": False
             , "filterByColor": False
             , "filterByArea": True
             , "minArea": 50.0
             , "maxArea": 250.0
             }
    disp = blobTracking \
        ( trblobs.trackBlobs \
            ( blob_params.mkDetector(params)
            , cupetri_half
            , lambda path: None if len(path) < 10 else flagger(path)
            , cropped
            , anchor_match_dist=20
            , max_flow_err=20
            , blur_size=4
            , debug = args.debug == 'tracking' and args.debug
            )
        )

    # TODO: include scale on analysis screen somehow
    with win():
        cviter.displaySink(windowName, disp, ending=True)

sentinel = \
    {
    }

default = \
    {
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
        ( '-b'
        , '--beginning'
        , metavar = 'i'
        , type = int
        , help = '''First frame to perform analysis on. (default: first frame, 0)''')
    p.add_argument \
        ( '-e'
        , '--ending'
        , metavar = 'j'
        , type = int
        , help = '''Last frame to perform analysis on. (default: last frame, dependent on the video)''')
    p.add_argument \
        ( '-c'
        , '--coin-diameter'
        , metavar = 'mm'
        , type = float
        , default = 19.05
        , help = '''Diameter of the coin in the frame in milimeters. (default: size of a US penny)''')
    p.add_argument \
        ( '-d'
        , '--debug'
        , metavar = 'stage'
        , help = '''To debug the system, give the name of a failing stage based on errors or warnings.''')

    # main
    exit(main(p.parse_args()))

# eof
