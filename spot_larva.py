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
import lib.color as color
import lib.util as util

def swatch(args):
    i, (color_name, color) = args
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58
    bottom_margin = 6
    left_margin = 4
    rows = 22
    rect_offset_cols = 70
    im = numpy.ndarray((rows, 192, 3), numpy.uint8)
    im.fill(255)
    cv2.rectangle(im, (rect_offset_cols, 0), (rect_offset_cols + rows, rows), color, -1)
    cv2.putText(im, 'Path ' + str(i), (left_margin, rows - bottom_margin), font, font_scale, 0)
    cv2.putText(im, color_name, (rect_offset_cols + rows + left_margin, rows - bottom_margin), font, font_scale, 0)
    return im

def annot_analysis(dst, src, paths):
    numpy.copyto(dst, src)
    for pth in paths:
        trblobs.annot_hist(dst, pth)

def annot_colors(dst, src, paths, colors):
    numpy.copyto(dst, src)
    numpy.copyto(dst, (dst * 0.25 + 255 * 0.75).astype(numpy.uint8))
    for pth, col in zip(paths, colors):
        trblobs.annot_hist(dst, pth, point=False, color=col, thickness=4)

def analysis_and_color_image(dst, src, paths):
    annot_analysis(dst[:,:src.shape[1]], src, paths)
    annot_colors(dst[:,src.shape[1]:], src, paths,
            map(color_lib.kariru, path_borrowers))

def blob_tracking(filepath, beginning, frame_count, flagger, stream, debug=None):
    span_count = frame_count - 1
    color_lib = color.ResourceLibrary({name: tuple(reversed(c)) for name, c in color.alphabet()})
    path_borrower = lambda path: id(path[0]) # FIXME: hax
    ns = None
    # span_i ranges inclusive over [0 .. len(stream) - 1]
    #   -> we have it start at 1, so it ranges over [1 .. len(stream)]
    # span_count is the number of between-frame moments
    for span_i, (fi, paths) in enumerate(stream, 1):
        print('> {}/{} tracking {} paths'.format(span_i, span_count, len(paths)))

        if ns is None:
            ns = collections.namedtuple('NS', 'secs previous_path_borrowers analysis colors out')\
                ( secs = set()
                , previous_path_borrowers = []
                , analysis = numpy.empty_like(fi.image)
                , colors = numpy.empty_like(fi.image)
                , out = numpy.hstack([numpy.empty_like(fi.image), numpy.empty_like(fi.image)])
                )

        t = max(pth[-1].frameinfo.msec - beginning.msec for pth in paths) / 1000
        sec = int(t)

        # generate analysis image
        annot_analysis(ns.analysis, fi.image, paths)

        # FIXME: there is a bug here somewhere with span_count; perhaps related to the different frame starting number on mac/linux
        if sec not in ns.secs or span_i == span_count:
            ns.secs.add(sec)

            # generate colors image
            outfile = '{}_result-T{T:.3}.png'.format(filepath, T=t)
            print('= Writing', outfile)
            filtered_paths = iterutils.remove(flagger, paths)
            # release unused colors
            path_borrowers = map(path_borrower, filtered_paths)
            map(color_lib.kaesu, set(ns.previous_path_borrowers) - set(path_borrowers))
            ns = ns._replace(previous_path_borrowers = path_borrowers)
            # make image
            annot_analysis(ns.out[:,:fi.image.shape[1]], fi.image, filtered_paths)
            annot_colors(ns.out[:,fi.image.shape[1]:], fi.image, filtered_paths,
                    map(color_lib.kariru, path_borrowers))
            cv2.imwrite(outfile, ns.out)

        yield ([ns.analysis], (paths, color_lib, path_borrowers, ns.out))


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
    return min(bucket.keys()), [bucket[T] for T in sorted(bucket)]

def blob_analysis \
        ( filepath
        , args
        , beginning
        , paths
        , mm_per_px = None
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

    fmt_s = lambda s: None if s is None else ';'.join('{:g}'.format(n) for n in s)

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
        , ['args', args]
        , ['{du}/{du_}'.format(du=du_name, du_=du_orig), mm_per_px]
        # +-------+--------------+
        # | file  | IMG_####.mov |
        # | args  | ...          |
        # | mm/px | #            |
        # +-------+--------------+

        , []

        ]
    d_rows = \
        [ ['Distance traveled ({du})'.format(du=du_name)]
        , group_header
        # +---+----------+----------+----------+----------+
        # | Distance traveled (mm)                        |
        # +---+----------+----------+----------+----------+
        # | P | T1 = 15" | T2 = 30" | T3 = 45" | T4 = 60" |
        # +---+----------+----------+----------+----------+
        ]
    cd_rows = \
        [ ['Cumulative distance traveled ({du})'.format(du=du_name)]
        , group_header
        # +---+----------+----------+----------+----------+
        # | Cumulative distance traveled (mm)             |
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
        [ ['Time bounds begin;end ({tu})'.format(tu=tu_name)]
        , group_header
        # +---+----------+----------+----------+----------+
        # | Time bounds begin;end (sec)                   |
        # +---+----------+----------+----------+----------+
        # | P | 0" - T1  | T1 - T2  | T2 - T3  | T3 - T4  |
        # +---+----------+----------+----------+----------+
        ]

    for P, path in enumerate(paths):

        # analyse path
        pad, pathlets = split_path(beginning, path)
        bounds = map(trblobs.path_time_bounds, pathlets)
        time = map(trblobs.path_elapsed_time, pathlets)
        dist = map(trblobs.path_dist, pathlets)
        cdist = reduce(lambda acc, cur: acc + [acc[-1] + cur], dist, [0])[1:]
        assert len(cdist) == len(dist)

        # assert correctness by comparing total-path analysis with sum of split-path analysis
        pbounds = trblobs.path_time_bounds(path)
        assert abs((pbounds[1] - pbounds[0]) - numpy.array([t1 - t0 for t0, t1 in bounds]).sum()) < 0.0001, \
                'time described by pathlet bounds must match that of overall bounds: P%d' % P
        assert abs(trblobs.path_elapsed_time(path) - numpy.array(time).sum()) < 0.0001, \
                'time described by pathlet must match total elapsed time: P%d' % P
        assert abs(trblobs.path_dist(path) - numpy.array(dist).sum()) < 0.0001, \
                'distance described by sumnation must match total distance: P%d' % P
        assert abs(trblobs.path_dist(path) - cdist[-1]) < 0.0001, \
                'distance described by cumulative sum must match total distance: P%d' % P

        # produce table data
        take = max(0, groups - pad)

        # +---+----------+----------+----------+----------+
        # | 0 |          |          |          |          |
        # | 1 |          |          |          |          |
        # | 2 |          |          |          |          |
        # +---+----------+----------+----------+----------+
        d_rows.append([P] + pad * [0] +
                map(du, dist[:take]))

        # +---+----------+----------+----------+----------+
        # | 0 |          |          |          |          |
        # | 1 |          |          |          |          |
        # | 2 |          |          |          |          |
        # +---+----------+----------+----------+----------+
        cd_rows.append([P] + pad * [0] +
                map(du, cdist[:take]))

        # +---+----------+----------+----------+----------+
        # | 0 |          |          |          |          |
        # | 1 |          |          |          |          |
        # | 2 |          |          |          |          |
        # +---+----------+----------+----------+----------+
        s_rows.append([P] + pad * [0] +
                [du(d) / tu(dt) for d, dt in zip(dist[:take], time)])

        # +---+----------+----------+----------+----------+
        # | 0 |          |          |          |          |
        # | 1 |          |          |          |          |
        # | 2 |          |          |          |          |
        # +---+----------+----------+----------+----------+
        b_rows.append([P] + pad * ['-'] +
                [fmt_s([tu(b - beginning.msec) for b in bs]) for bs in bounds[:take]])

    return g_rows + [[]] + d_rows + [[]] + cd_rows + [[]] + s_rows + [[]] + b_rows

def write_table(outfile, table_data):
    with open(outfile, mode='wb') as fd:
        writer = csv.writer(fd, dialect='excel')
        for r in table_data:
            writer.writerow(r)
    print('csv written:', outfile)


class QueryAOI(object):

    @classmethod
    def query_aoi_main(cls, filepath, codeword, windower, raw_frames, cached=False, annotator=None):
        '''str, str, WindowMaker, iter<numpy.array>[, bool][, MouseQuery-annotator] -> AOI, iter<numpy.array>

           Load the AOI for `codeword` from the cache for `filepath` or fall back querying the user.
           Crop the stream of frames.
        '''
        annot = cls.annot_bqr if annotator is None else annotator
        if cached:
            # read cache or ask user, conditionally save cache
            aoiM = cls.query_aoi_cache(filepath, codeword)
            if aoiM is None:
                aoi_frames = cls.ask_for_aoi(windower, raw_frames, annot)
                cls.save_aoi_cache(filepath, codeword, aoi_frames[0])
                return aoi_frames
            else:
                return aoiM, cls.apply_aoi(aoiM, raw_frames)
        else:
            return cls.ask_for_aoi(windower, raw_frames, annot)

    @classmethod
    def ask_for_aoi(cls, windower, raw_frames, annotator):
        '''WindowMaker, iter<numpy.array>, MouseQuery-annotator -> Box, iter<numpy.array>

           Query the user for their area of interest. Return it and a stream of cropped frames.
        '''
        # TODO: add a question for the user to the screen "Click to draw a box around {codeword}."
        first = next(raw_frames)
        frames = itertools.chain([first], raw_frames)
        # query for a general area on the frame
        with windower as ow:
            with mouse.MouseQuery \
                    ( ow
                    , first
                    , point_count = 2
                    , annot_fn = annotator
                    ) as query:
                pts = query()
        x0, x1 = sorted(max(col, 0) for col,_ in pts)
        y0, y1 = sorted(max(row, 0) for _,row in pts)
        aoi = util.Box(util.Point2D(x=x0, y=y0), util.Point2D(x=x1, y=y1))
        return aoi, cls.apply_aoi(aoi, frames)

    @classmethod
    def query_aoi_cache(cls, filepath, codeword):
        '''str, str -> maybe<Box>'''
        try:
            with open(cls.aoi_cache_name(filepath, codeword), 'rb') as npy:
                return util.Box.from_rc_arr(numpy.load(npy)) # Cache hit, return $ Just aoi
        except IOError as e:
            if e.errno == 2:
                return # Cache miss, return Nothing
            else:
                raise e # Bug, re-raise

    @classmethod
    def save_aoi_cache(cls, filepath, codeword, aoi):
        '''str, str, Box -> None'''
        with open(cls.aoi_cache_name(filepath, codeword), 'wb') as npy:
            numpy.save(npy, aoi.pt_rc_arr)

    @staticmethod
    def aoi_cache_name(filepath, codeword):
        '''str, str -> str'''
        return '{}.{}.npy'.format(filepath, codeword)

    @staticmethod
    def apply_aoi(aoi, raw_frames):
        '''AOI, iter<numpy.array> -> iter<numpy.array>'''
        s = aoi.slices
        return itertools.imap(lambda im: im[s], raw_frames)

    @staticmethod
    def annot_bqr(*args):
        '''a MouseQuery annotator which facilitates the selection of boxes'''
        _, lmb, _, _ = args
        mouse.annotate_box(*args, color_fn=lambda *_: (0,255,255))
        if lmb:
            mouse.annotate_quadrants(*args)
        mouse.annotate_reticle(*args, color_fn=lambda *_: (0,0,255), size=35, thickness=3)


class CircleForScale(object):

    @staticmethod
    def circle_for_scale_main(codeword, frames, min_ct, max_std, circle_iter_kwargs, diameter_mm=None, debug=None):
        '''... -> maybe<(ndarray<3>, ndarray<3>, maybe<float>)>

           Attempt to detect a circle in the stream of frames.

           When no circle is detected, return None.
           Otherwise, return (mean_x_y_radius, std_x_y_radius, mm_per_px).
           * The `mm_per_px` result is only included when `diameter_mm` is given.
        '''
        if diameter_mm is not None:
            assert diameter_mm > 0

        circleM = circles.find_circle \
            ( codeword
            , min_ct
            , max_std
            , frames
            , debug = debug == codeword and debug
            , **circle_iter_kwargs
            )

        if circleM is None:
            print('! Error: No %s was not detected.' % codeword.title())
            print('!    Try re-running with `-d/--debug %s` to see what is failing.' % codeword)
            return
        else:
            pct, mean, std = circleM
            mean_ctr, mean_rad = mean[:2], mean[2]
            std_ctr,   std_rad =  std[:2],  std[2]
            mean_diam = 2 * mean_rad
            std_diam  = 2 *  std_rad

        if pct < 0.9:
            print('! Warning: %s was detected in less than 90%% of frames.' % codeword.title())
            print('!    Try re-running with `-d/--debug %s` to see what is failing.' % codeword)

        if std_rad >= 0.1 * mean_rad:
            print('! Warning: %s radius standard deviation is >= 10%% of radius mean.' % codeword.title())
            print('!    Try re-running with `-d/--debug %s` to see what is failing.' % codeword)

        if diameter_mm is None:
            scale = 1
            unit = 'px'
        else:
            mm_per_px = diameter_mm / mean_diam
            print('= Scale from {}: {:g} mm/px'.format(codeword, mm_per_px))
            scale = mm_per_px
            unit = 'mm'

        print('= Mean {} diameter: {:g}{u} (standard deviation: {:g}{u})'.
            format(codeword, mean_diam * scale, std_diam * scale, u=unit))

        return (mean, std, mm_per_px if diameter_mm else None)

    @staticmethod
    def annot_coin_result(dst, box, circle):
        box.rectangle(dst, (0,255,255))
        circles.annot_target(int(circle[0]), int(circle[1]), int(circle[2]), dst)

    @classmethod
    def mk_annot_bqrc(cls, minFraction=None, maxFraction=None, **_):
        def annot_bqrc(*args):
            QueryAOI.annot_bqr(*args)
            dst, lmb, xy, pts = args
            if pts:
                box = numpy.array([pts[-1], xy])
                cls.annot_circle_extents \
                    ( dst
                    , tuple(box.mean(axis=0).round().astype(int))
                    , abs(box[1] - box[0]).min() // 2
                    , minFraction
                    , maxFraction
                    )
        return annot_bqrc

    @staticmethod
    def annot_circle_extents(dst, pt, r, minFraction, maxFraction):
        # make green circle 20% smaller to encourage selection of a box that contains the coin edges
        cv2.circle(dst, pt, int(r * 0.8), (0,255,0), 2)
        cv2.circle(dst, pt, int(r * minFraction), (0,255,255), 2)
        cv2.circle(dst, pt, int(r * maxFraction), (0,255,255), 1)


# images : iter<frameinfo> -> iter<numpy.array>
images = functools.partial(itertools.imap, lambda fi: fi.image)


def applyto_images(frameinfos, fn):
    '''iter<FrameInfo>, [iter<numpy.array> -> iter<numpy.array>] -> iter<FrameInfo>

       Same as cviter.applyTo, but specified to operate only on the image component of frameinfos.
    '''
    return cviter.applyTo \
        ( lambda fi: fi.image
        , lambda fi, im: fi._replace(image=im)
        , fn # functools.partial(itertools.imap, fn)
        , frameinfos
        )


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


# TODO: move this into AOI.from_circle?
# TODO: make a Circle class like AOI to hold all the relevant ideas
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

    source_pathroot, _ = os.path.splitext(args.movie.source)

    window_maker = cvutils.WindowMaker \
        ( '{0} - {1}'.format(os.path.basename(sys.argv[0]), args.movie.source)
        , flags = cv2.WINDOW_NORMAL
        , width_height = None if args.window_height is None else (int(args.window_height * 4 / 3), args.window_height)
        )

    cue = lambda step=None: args.movie[args.beginning:args.ending:step]
    cue_frame_count = (args.ending or args.movie.frame_count) - (args.beginning or 0)
    first_frame = args.movie[args.beginning or 0]
    debug_image = first_frame.image.copy()

    # (scale1) coin for scale
    # in:
    #   ?
    # out:
    #   mm_per_px_coin (used by scale3)
    if args.coin:
        coin_iter_conf = dict \
            ( blur = 8
            , param2 = 25
            , minFraction = 0.5
            , maxFraction = 1.4
            )
        # TODO: must print question on the frame somewhere
        coin_aoi, coin_frames \
            = QueryAOI.query_aoi_main \
            ( source_pathroot
            , 'coin'
            , window_maker
            , images(cue(step=3))
            , cached = True
            , annotator = CircleForScale.mk_annot_bqrc(**coin_iter_conf)
            )
        if args.only_coin:
            print('= Terminating after collecting (or verifying) coin area-of-interest')
            print(coin_aoi)
            return 0 # early termination from main
        # TODO: write magic numbers in terms of frame resolution or appropriate
        coin_mean_rel, _, mm_per_px_coin \
            = CircleForScale.circle_for_scale_main \
            ( 'coin'
            , coin_frames
            , 15   # min_ct
            , 3.25 # max_std
            , coin_iter_conf
            , diameter_mm = args.coin_diameter
            , debug = args.debug
            )
        CircleForScale.annot_coin_result \
            ( debug_image
            , coin_aoi
            , numpy.concatenate([coin_aoi.pt0.xy + coin_mean_rel[:2], coin_mean_rel[2:]])
            )

    # TODO: write magic numbers in terms of frame resolution or appropriate
    # (scale2) petri dish for crop
    # in:
    #   ?
    # out:
    #   petri_mean (used several places, including for cropping the image)
    #   mm_per_px_petri (used by scale3)
    petri_mean, _, mm_per_px_petri = CircleForScale.circle_for_scale_main \
        ( 'petri'
        , images(cue(step=3))
        , 10   # min_ct
        , 15.0 # max_std
        , dict \
            ( blur = 10
            , param2 = 50
            , minFraction = 0.8
            , maxFraction = 1.2
            )
        , diameter_mm = args.petri_dish_diameter
        , debug = args.debug
        )
    circles.annot_target(int(petri_mean[0]), int(petri_mean[1]), int(petri_mean[2]), debug_image)

    # (scale3) compute scale for later data generation
    # in:
    #   pp_per_px_petri
    #   [mm_per_px_coin]
    # out:
    #   mm_per_px (used several places)
    if args.coin:
        mm_per_px = (mm_per_px_coin + mm_per_px_petri) / 2
        print('= Average scale {:g} mm/px'.format(mm_per_px))
    else:
        mm_per_px = mm_per_px_petri
        print('! Warning: Using only petri dish for scale because --no-coin was given.')

    # save debug image w/ coin & petri dish
    cv2.imwrite(source_pathroot + '_result-circles.png', debug_image)

    # TODO: clean up these- use box, or make a circle class?
    _, ((cbbx, cbby), (cbbw, cbbh)), cc = circle_bb \
        ( petri_mean
        , ( args.movie.frame_width
          , args.movie.frame_height
          )
        )
    cupetri = numpy.concatenate((cc, petri_mean[2:]))
    cupetri_half = numpy.concatenate((cupetri[:2], 0.5 * cupetri[2:]))

    # crop to the petri-dish circle
    croppedA, croppedB = itertools.tee \
        ( applyto_images \
            ( cue()
            , lambda upstream: QueryAOI.apply_aoi \
                ( util.Box \
                    ( util.Point2D(x=cbbx, y=cbby)
                    , util.Point2D(x=cbbx+cbbw, y=cbby+cbbh)
                    )
                , mask_circle(petri_mean, upstream)
                )
            )
        )

    # track
    flagger = trblobs.gen_flagger(cupetri_half, args.max_flow_fraction)
    # params still set to their opencv defaults:
    #   minRepeatability
    #   thresholdStep
    #   minDistBetweenBlobs
    params = dict \
        ( filterByConvexity = False
        , filterByCircularity = False
        , filterByInertia = False
        , filterByColor = False
        , filterByArea = True
        , minArea = float(args.min_blob_area / mm_per_px)
        , maxArea = float(args.max_blob_area / mm_per_px)
        # detect min/max threshold from image brightness?
        , minThreshold =  50.0
        , maxThreshold = 250.0
        , thresholdStep = 5.0
        )
    disp = blob_tracking \
        ( source_pathroot
        , first_frame
        , cue_frame_count
        , flagger
        , itertools.izip \
            ( itertools.islice(croppedA, 1, None)
            , trblobs.trackBlobs \
                ( blob_params.mkDetector(params)
                , cupetri_half
                , flagger
                , croppedB
                , anchor_match_dist = args.anchor_match_dist / mm_per_px
                , max_flow_err = 20
                # detect blur_size from mm_per_px?
                , blur_size = 4
                , debug = args.debug == 'tracking' and args.debug
                )
            )
        )

    # consume the stream & run final analysis
    with window_maker as window:
        ret = cviter.displaySink(window, disp, ending=False)
    _, color_lib, path_borrowers, final_image = ret.result
    paths = iterutils.remove(flagger, ret.result[0])
    assert len(paths) == len(path_borrowers), 'filtered paths correspond to borrowers'

    # save color legend
    color_legend = numpy.concatenate(map(swatch,
        enumerate(map(color_lib.query, path_borrowers))), axis=0)
    cv2.imwrite('{}_result-colors.png'.format(source_pathroot), color_legend)

    print('= Fully consumed' if ret.fully_consumed else '= Early termination')
    print('= {} paths'.format(len(paths)))

    # TODO: Consider outputting the raw path data as essential state to a pickle format and then only producing a table once different analysis methods have been concieved
    table_data = blob_analysis \
        ( args.movie.source
        , args
        , first_frame
        , paths
        , mm_per_px = mm_per_px
        , )

    if ret.fully_consumed:
        write_table(source_pathroot + '_result.csv', table_data)
    else:
        print('table_data', table_data)

    return 0 # great success

sentinel = \
    {
    }

default = \
    { 'flow_fraction' : 0.4
    , 'min_blob_area': 10.0
    , 'max_blob_area': 35.0
    , 'anchor_match_dist': 3.0
    }

def flow_fraction(f_str):
    f = float(f_str)
    if 0 <= f <= 1:
        return f
    else:
        raise argparse.ArgumentTypeError('flow fraction is a value between 0 and 1, incl')

if __name__ == '__main__':

    # test
    doctests = map(lambda m: (m, doctest.testmod(m)),
        [ None
        , blob_params
        , circles
        , color
        , cviter
        , cvutils
        , iterutils
        , mouse
        , trblobs
        , util
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
        ( '-p'
        , '--petri-dish-diameter'
        , metavar = 'mm'
        , type = float
        , default = 15 * 10
        , help = '''Diameter of the petri dish in the frame in millimeters. (default: 15mm)''')
    p.add_argument \
        ( '-f'
        , '--max-flow-fraction'
        , metavar = 'f'
        , type = flow_fraction
        , default = default['flow_fraction']
        , help = '''Allowed portion of a path which is optical-flow. (default: {:g})'''.format(default['flow_fraction']))
    p.add_argument \
        ( '--min-blob-area'
        , metavar = 'mm^2'
        , type = float
        , default = default['min_blob_area']
        , help = '''Minimum area of blobs detected by the SimpleBlobDetector. (default {:g}mm^2)'''.format(default['min_blob_area']))
    p.add_argument \
        ( '--max-blob-area'
        , metavar = 'mm^2'
        , type = float
        , default = default['max_blob_area']
        , help = '''Maximum area of blobs detected by the SimpleBlobDetector. (default {:g}mm^2)'''.format(default['max_blob_area']))
    p.add_argument \
        ( '--anchor-match-dist'
        , metavar = 'mm'
        , type = float
        , default = default['anchor_match_dist']
        , help = '''Maximum allowed distance to match a path head & a detected blob. (default {:g}mm)'''.format(default['anchor_match_dist']))
    p.add_argument \
        ( '-d'
        , '--debug'
        , metavar = 'stage'
        , help = '''To debug the system, give the name of a failing stage based on errors or warnings.''')
    p.add_argument \
        ( '--no-coin'
        , dest='coin'
        , action = 'store_false'
        , help = '''If there is no coin in the movie, give this option to skip searching for one (and rely solely on petri-dish size).''')
    p.add_argument \
        ( '--only-coin'
        , action = 'store_true'
        , help = '''Cause the program to exit after user interaction & caching of the coin's location in the frame.''')
    p.add_argument \
        ( '-wh'
        , '--window-height'
        , metavar = 'px'
        , type = int
        , help = '''Resize the window to a 4:3 landscape with this height. Default behavior is operating system dependent (Linux fits the screen, OSX doesn't).''')

    # main
    exit(main(p.parse_args()))

# eof
