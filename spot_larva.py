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
import os.path
import functools
import itertools
import argparse
import doctest
import contextlib
import csv

import numpy
import cv2

import lib.track_blobs as trblobs
import lib.blob_params as blob_params
import lib.mouse as mouse
import lib.circles as circles
import lib.cviter as cviter
import lib.cvutils as cvutils
import lib.iterutils as iterutils

def blob_tracking(length, stream, debug=None):
    # TODO: figure out why the i enumerating over paths is off by 2
    #   length is the frame count, but i is zero indexed
    #   stream yields once for each pair of frames
    # TODO: do annotation here instead of trblobs.trackBlobs
    # TODO: compose the penny into the top left of the frames before they are displayed
    for i, (annotCur, annotHist, paths) in enumerate(stream):
        print('> At frame {}/{} tracking {} paths'.format(i, length, len(paths)))
        yield ([annotCur, annotHist], paths)

def split_path(beginning, path):
    '''FrameInfo, Path -> [Path]

       Split path up by the number of seconds elapsed since the beginning of analysis.

       For each bucketed pathlet, T, prepend the final point from bucket T - 1, if any.
       (This prevents incorrect time and distance *span* calculations.)
    '''
    # bucket points by second since beginning
    bucket = collections.defaultdict(collections.deque)
    for point in path:
        bucket[int((point.frameinfo.msec - beginning.msec) // 1000)].append(point)
    bucket.default_factory = None
    # prepend final point of each bucket onto the next bucket
    for T in sorted(bucket)[1:]: # keys except for 0
        assert T > 0
        bucket[T].appendleft(bucket[T - 1][-1])
    return [bucket[T] for T in sorted(bucket)]

def blob_analysis \
        ( filepath
        , beginning
        , paths
        , upetri = None
        , spetri = None
        , mm_per_px = None
        , ucoin = None
        , scoin = None
        , bbcoin = None
        ):
    print('= Path summary')
    for i, p in enumerate(paths):
        print \
            ( 'P%d' % i
            , '\t'
            , '{:g},{:g}'.format(*p[0].pt)
            , '-{}-nodes->'.format(len(p))
            , '{:g},{:g}'.format(*p[-1].pt)
            )

    fmt_s = lambda s: None if s is None else ','.join('{:g}'.format(n) for n in s)

    du_orig = 'px'
    du_name = 'px' if mm_per_px is None else 'mm'
    du_per_px = 1 if mm_per_px is None else mm_per_px
    du = lambda px: px * du_per_px

    tu_name = 'sec'
    fastsec_per_sec = 15
    tu = lambda msec: msec / 1000 * fastsec_per_sec

    groups = 4
    # | P | T1 = 15" | T2 = 30" | T3 = 45" | T4 = 60" |
    group_header = [ 'P' ] + ['T{:d} = {:d}"'.format(fastsec, fastsec * fastsec_per_sec) for fastsec in xrange(1, 1 + groups)]

    # produce global table data
    g_rows = \
        [ ['file', os.path.split(filepath)[1]]
        , ['{du}/{du_}'.format(du=du_name, du_=du_orig), mm_per_px]
        # +-------+--------------+
        # | file  | IMG_####.mov |
        # | mm/px | #            |
        # +-------+--------------+

        , []

        , ['object', 'mean x,y,r ({du_})'.format(du_=du_orig), 'std x,y,r ({du_})'.format(du_=du_orig), 'bb']
        , ['coin', fmt_s(ucoin), fmt_s(scoin), fmt_s(bbcoin)]
        , ['petri dish', fmt_s(upetri), fmt_s(spetri)]
        # +-------+-----------------+----------------+---------------------+
        # | obj   | mean x,y,r (px) | std x,y,r (px) | bb (x0, x1, y0, y1) |
        # +-------+-----------------+----------------+---------------------+
        # | coin  | #,#,#           | #,#,#          | #,#,#,#             |
        # | petri | #,#,#           | #,#,#          | #,#,#,#             |
        # +-------+-----------------+----------------+---------------------+
        ]
    d_rows = \
        [ ['Distance traveled ({du})'.format(du=du_name)]
        , group_header
        # +---+----------+----------+----------+----------+
        # | Distance traveled (mm)               |
        # +---+----------+----------+----------+----------+
        # | P | T1 = 15" | T2 = 30" | T3 = 45" | T4 = 60" |
        # +---+----------+----------+----------+----------+
        ]
    s_rows = \
        [ ['Average speed ({du}/{tu})'.format(du=du_name, tu=tu_name)]
        , group_header
        # +---+----------+----------+----------+----------+
        # | Average speed (mm/sec)            |
        # +---+----------+----------+----------+----------+
        # | P | 0" - T1  | T1 - T2  | T2 - T3  | T3 - T4  |
        # +---+----------+----------+----------+----------+
        ]
    b_rows  = \
        [ ['Time bounds ({tu},{tu})'.format(tu=tu_name)]
        , group_header
        # +---+----------+----------+----------+----------+
        # | Time bounds (sec,sec)                |
        # +---+----------+----------+----------+----------+
        # | P | 0" - T1  | T1 - T2  | T2 - T3  | T3 - T4  |
        # +---+----------+----------+----------+----------+
        ]

    for P, path in enumerate(paths):

        # analyse path
        pathlets = split_path(beginning, path)
        bounds = map(trblobs.path_time_bounds, pathlets)
        time = map(trblobs.path_elapsed_time, pathlets)
        dist = map(trblobs.path_dist, pathlets)

        # assert correctness by comparing total-path analysis with sum of split-path analysis
        pbounds = trblobs.path_time_bounds(path)
        assert abs((pbounds[1] - pbounds[0]) - numpy.array([t1 - t0 for t0, t1 in bounds]).sum()) < 0.0001, \
                'time described by pathlet bounds must match that of overall bounds: P%d' % P
        assert abs(trblobs.path_elapsed_time(path) - numpy.array(time).sum()) < 0.0001, \
                'time described by pathlet must match total elapsed time: P%d' % P
        assert abs(trblobs.path_dist(path) - numpy.array(dist).sum()) < 0.0001, \
                'distance described by sumnation must match total distance: P%d' % P

        # produce table data
        pad = groups - len(pathlets)

        # +---+----------+----------+----------+----------+
        # | 0 |          |          |          |          |
        # | 1 |          |          |          |          |
        # | 2 |          |          |          |          |
        # +---+----------+----------+----------+----------+
        d_rows.append([P] + pad * ['-'] +
                map(du, dist[:groups]))

        # +---+----------+----------+----------+----------+
        # | 0 |          |          |          |          |
        # | 1 |          |          |          |          |
        # | 2 |          |          |          |          |
        # +---+----------+----------+----------+----------+
        s_rows.append([P] + pad * ['-'] +
                [du(d) / tu(dt) for d, dt in zip(dist[:groups], time)])

        # +---+----------+----------+----------+----------+
        # | 0 |          |          |          |          |
        # | 1 |          |          |          |          |
        # | 2 |          |          |          |          |
        # +---+----------+----------+----------+----------+
        b_rows.append([P] + pad * ['-'] +
                map(lambda b: fmt_s(map(tu, b)), bounds[:groups]))

    return g_rows + [[]] + d_rows + [[]] + s_rows + [[]] + b_rows

def write_table(outfile, table_data):
    with open(outfile, mode='wb') as fd:
        writer = csv.writer(fd, dialect='excel')
        for r in table_data:
            writer.writerow(r)
    print('csv written:', outfile)

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
    print('= Mean coin diameter: {:g}px (standard deviation: {:g}px)'.format(ud, sr))
    mm_per_px = coin_diameter_mm / ud
    print('= Scale: {:g} mm/px'.format(mm_per_px))
    return mm_per_px, [x0, y0, 0] + mean, std, (x0, x1, y0, y1)

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
    print('= Mean petri dish diameter: {:g}px (standard deviation: {:g}px)'.format(ud, sr))
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
    windowName = '{0} - {1}'.format(os.path.basename(sys.argv[0]), args.movie.source)
    win = functools.partial(window, windowName, cv2.WINDOW_NORMAL)

    # movie (source)
    cue = lambda step=None: args.movie[args.beginning:args.ending:step]
    cue_length = (args.ending or args.movie.frame_count) - (args.beginning or 0)

    # coin for scale
    # TODO: must print question on the frame somewhere
    if args.no_coin:
        mm_per_px, ucoin, scoin, bbcoin = None, None, None, None
    else:
        mm_per_px, ucoin, scoin, bbcoin = coin_for_scale(windowName, args.coin_diameter, cue(step=3), debug=args.debug)

    # petri dish for crop
    upetri, spetri = petri_for_crop(cue(step=3), debug=args.debug)
    if mm_per_px is not None:
        print('= Petri dish diameter is {:g}cm'.format(upetri[2] * 2 * mm_per_px / 10))
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
    disp = blob_tracking \
        ( cue_length
        , trblobs.trackBlobs \
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

    # consume the stream & run final analysis
    with win():
        ret = cviter.displaySink(windowName, disp, ending=False)
    paths = filter(lambda p: not flagger(p), ret.result)
    print('= Fully consumed' if ret.fully_consumed else '= Early termination')
    print('= {} paths'.format(len(paths)))

    # TODO: Consider outputting the raw path data as essential state to a pickle format and then only producing a table once different analysis methods have been concieved
    table_data = blob_analysis \
        ( args.movie.source
        , args.movie[args.beginning or 0]
        , paths
        , upetri = upetri
        , spetri = spetri
        , mm_per_px = mm_per_px
        , ucoin = ucoin
        , scoin = scoin
        , bbcoin = bbcoin
        , )

    if ret.fully_consumed:
        write_table(args.movie.source + '.csv', table_data)
    else:
        print('table_data', table_data)

    return 0 # great success

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
        , help = '''Diameter of the coin in the frame in millimeters. (default: size of a US penny)''')
    p.add_argument \
        ( '-d'
        , '--debug'
        , metavar = 'stage'
        , help = '''To debug the system, give the name of a failing stage based on errors or warnings.''')
    p.add_argument \
        ( '--no-coin'
        , action = 'store_true'
        , help = '''If there is no coin in the movie, give this option to skip scaling the data to millimeters.''')

    # main
    exit(main(p.parse_args()))

# eof
