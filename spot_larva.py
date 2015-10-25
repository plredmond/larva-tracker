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

def blob_tracking(filepath, beginning, length, stream, debug=None):
    # TODO: figure out why the i enumerating over paths is off by 2
    #   length is the frame count, but i is zero indexed
    #   stream yields once for each pair of frames
    # TODO: do annotation here instead of trblobs.trackBlobs
    # TODO: compose the penny into the top left of the frames before they are displayed
    secs = set()
    for i, (annotCur, annotHist, paths) in enumerate(stream):
        print('> At frame {}/{} tracking {} paths'.format(i, length, len(paths)))
        s = int(max(map(lambda path: path[-1].frameinfo.msec - beginning.msec, paths)) // 1000)
        if s not in secs:
            secs.add(s)
            cv2.imwrite(filepath + '_T{T}-hist.png'.format(T=s), annotHist)
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

        , [ 'object'
          , 'mean x;y;r ({du_})'.format(du_=du_orig)
          , 'std x;y;r ({du_})'.format(du_=du_orig)
          , 'bb x0;y0;x1;y1 (px)'
          ]
#       , ['coin', fmt_s(ucoin), fmt_s(scoin), fmt_s(None if bbcoin is None else (bbcoin[0] + bbcoin[1]))]
#       , ['petri dish', fmt_s(upetri), fmt_s(spetri)]
        # +-------+-----------------+----------------+---------------------+
        # | obj   | mean x;y;r (px) | std x;y;r (px) | bb x0;y0;x1;y1 (px) |
        # +-------+-----------------+----------------+---------------------+
        # | coin  | #;#;#           | #;#;#          | #;#;#;#             |
        # | petri | #;#;#           | #;#;#          | #;#;#;#             |
        # +-------+-----------------+----------------+---------------------+
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
                [fmt_s([tu(b - beginning.msec) for b in bs]) for bs in bounds[:groups]])

    return g_rows + [[]] + d_rows + [[]] + s_rows + [[]] + b_rows

def write_table(outfile, table_data):
    with open(outfile, mode='wb') as fd:
        writer = csv.writer(fd, dialect='excel')
        for r in table_data:
            writer.writerow(r)
    print('csv written:', outfile)


class AOI(object):
    __slots__ = '__arr'
    def __init__(self, pts):
        '''[[i0,j0,k0, ...], [i1,j1,k1, ...]] -> AOI'''
        assert pts.ndim == 2, 'AOI: pts must have two dimensions'
        assert pts.shape[0] == 2, 'AOI: must contain 2 points'
        assert (pts[0] < pts[1]).all(), 'AOI: points must represent a box'
        self.__arr = pts.copy()
    @property
    def ptarr(self):
        '''AOI -> numpy.array([[i0,j0,k0], [i1,j1,k1]])'''
        return self.__arr.copy()
    @property
    def pts(self):
        '''AOI -> ((i0,j0,k0), (i1,j1,k1))'''
        return tuple(map(tuple, self.__arr))
    @property
    def pt0_size(self):
        '''AOI -> ((i0,j0,k0), (i1-i0,j1-j0,k1-k0))'''
        pt0, pt1 = self.__arr
        return tuple(pt0), tuple(pt1 - pt0)
    @property
    def slices(self):
        '''AOI -> (slice(i0,i1), slice(j0,j1), slice(k0,k1))'''
        return tuple(map(functools.partial(apply, slice), self.__arr.transpose()))

    def rectangle(self, img, color, **kwargs):
        '''numpy.ndarray, tup<num>[, kwargs of cv2.rectangle] -> None'''
        pt0, pt1 = self.pts
        return cv2.rectangle(img, tuple(reversed(pt0)), tuple(reversed(pt1)), color, **kwargs)
    def crop(self, img):
        '''numpy.ndarray -> numpy.ndarray'''
        return img[self.slices]
    def expand(self, n):
        '''num -> AOI'''
        pts = self.ptarr
        pts[0] -= n
        pts[1] += n
        return type(self)(pts)

    def __repr__(self):
        return '{}(\n{}\n)'.format(type(self).__name__, repr(self.__arr))

    @classmethod
    def from_pts(cls, pts):
        '''((i0,j0,k0), (i1,j1,k1)) -> AOI'''
        return cls(numpy.array(pts))
    @classmethod
    def from__pt0_pt1(cls, pt0, pt1):
        '''(i0,j0,k0), (i1,j1,k1) -> AOI'''
        return cls.from_pts((pt0, pt1))
    @classmethod
    def from__pt0_size(cls, base_size):
        '''((i0,j0,k0), (i1-i0,j1-j0,k1-k0)) -> AOI'''
        base, size = base_size
        return cls.from__pt0_pt1(base, numpy.array(base) + size)


class QueryAOI(object):

    @classmethod
    def query_aoi_main(cls, filepath, codeword, windower, raw_frames, cached=False):
        '''str, str, WindowMaker, iter<numpy.array>[, bool] -> AOI, iter<numpy.array>

           Load the AOI for `codeword` from the cache for `filepath` or fall back querying the user.
           Crop the stream of frames.
        '''
        if cached:
            # read cache or ask user, conditionally save cache
            aoiM = cls.query_aoi_cache(filepath, codeword)
            if aoiM is None:
                aoi_frames = cls.ask_for_aoi(windower, raw_frames)
                cls.save_aoi_cache(filepath, codeword, aoi_frames[0])
                return aoi_frames
            else:
                return aoiM, cls.apply_aoi(aoiM, raw_frames)
        else:
            return cls.ask_for_aoi(windower, raw_frames)

    @classmethod
    def ask_for_aoi(cls, windower, raw_frames):
        '''WindowMaker, iter<numpy.array> -> AOI, iter<numpy.array>

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
                    , annot_fn = cls.annot_bqr
                    ) as query:
                pts = query()
        col0, col1 = sorted(max(col, 0) for col,_ in pts)
        row0, row1 = sorted(max(row, 0) for _,row in pts)
        aoi = AOI.from__pt0_pt1((row0, col0), (row1, col1))
        return aoi, cls.apply_aoi(aoi, frames)

    @classmethod
    def query_aoi_cache(cls, filepath, codeword):
        '''str, str -> maybe<AOI>'''
        try:
            with open(cls.aoi_cache_name(filepath, codeword), 'rb') as npy:
                return AOI(numpy.load(npy)) # Cache hit, return $ Just aoi
        except IOError as e:
            if e.errno == 2:
                return # Cache miss, return Nothing
            else:
                raise e # Bug, re-raise

    @classmethod
    def save_aoi_cache(cls, filepath, codeword, aoi):
        '''str, str, AOI -> None'''
        with open(cls.aoi_cache_name(filepath, codeword), 'wb') as npy:
            numpy.save(npy, aoi.ptarr)

    @staticmethod
    def aoi_cache_name(filepath, codeword):
        '''str, str -> str'''
        return '{}_{}.npy'.format(filepath, codeword)

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
        mouse.annotate_reticle(*args, color_fn=lambda *_: (0,0,255), size=25)


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
            ( min_ct
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
    def debug_image(src, pts, mean, expand=4):
        dst = src.copy()
        cv2.rectangle(dst, pts[0], pts[1], (0,255,255))
        circles.annot_target(int(mean[0]), int(mean[1]), int(mean[2]), dst)
        return dst


# images : iter<frameinfo> -> iter<numpy.array>
images = functools.partial(itertools.imap, lambda fi: fi.image)


def applyto_images(frameinfos, fn):
    '''iter<FrameInfo>, [iter<numpy.array> -> iter<numpy.array>] -> iter<FrameInfo>'''
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

    # coin for scale
    if not args.coin:
        mm_per_px_coin = None
    else:
        # TODO: must print question on the frame somewhere
        coin_aoi, coin_frames \
            = QueryAOI.query_aoi_main \
            ( source_pathroot
            , 'coin'
            , window_maker
            , images(cue(step=3))
            , cached = True
            )
        # TODO: write magic numbers in terms of frame resolution or appropriate
        coin_mean_relative, _, mm_per_px_coin \
            = CircleForScale.circle_for_scale_main \
            ( 'coin'
            , coin_frames
            , 15   # min_ct
            , 3.25 # max_std
            , dict \
                ( blur = 8
                , param2 = 25
                , minFraction = 0.5
                , maxFraction = 1.5
                )
            , diameter_mm = args.coin_diameter
            , debug = args.debug
            )
        cv2.imwrite \
            ( '{}_{}.png'.format(source_pathroot, 'coin')
            , CircleForScale.debug_image \
                ( first_frame.image
                , map(lambda p: tuple(reversed(p)), coin_aoi.pts)
                , numpy.concatenate([tuple(reversed(coin_aoi.pts[0])), [0]]) + coin_mean_relative
                )
            )

    # petri dish for crop
    # TODO: write magic numbers in terms of frame resolution or appropriate
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
    _, ((cbbx, cbby), (cbbw, cbbh)), cc = circle_bb \
        ( petri_mean
        , ( args.movie.frame_width
          , args.movie.frame_height
          )
        )
    cupetri = numpy.concatenate((cc, petri_mean[2:]))
    cupetri_half = numpy.concatenate((cupetri[:2], 0.5 * cupetri[2:]))

    mm_per_px = (mm_per_px_coin + mm_per_px_petri) / 2
    print('= Average scale {:g} mm/px'.format(mm_per_px))

    croppedA, croppedB = itertools.tee \
        ( applyto_images \
            ( cue()
            , lambda upstream: \
                QueryAOI.apply_aoi(AOI.from__pt0_size([(cbby, cbbx), (cbbh, cbbw)]),
                    mask_circle(petri_mean, upstream))
            )
        )

    # track
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
        ( args.movie.source
        , first_frame
        , cue_frame_count
        , flagger
        , itertools.izip \
            ( itertools.islice(croppedA, 1, None)
            , trblobs.trackBlobs \
                ( blob_params.mkDetector(params)
                , cupetri_half
                , lambda path: None if len(path) < 10 else flagger(path)
                , croppedB
                , anchor_match_dist=20
                , max_flow_err=20
                , blur_size=4
                , debug = args.debug == 'tracking' and args.debug
                )
            )
        )

    # consume the stream & run final analysis
    with window_maker as window:
        ret = cviter.displaySink(window, disp, ending=False)
    paths = filter(lambda p: not flagger(p), ret.result)
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
        ( '-p'
        , '--petri-dish-diameter'
        , metavar = 'mm'
        , type = float
        , default = 15 * 10
        , help = '''Diameter of the petri dish in the frame in millimeters. (default: 15mm)''')
    p.add_argument \
        ( '-d'
        , '--debug'
        , metavar = 'stage'
        , help = '''To debug the system, give the name of a failing stage based on errors or warnings.''')
    p.add_argument \
        ( '--no-coin'
        , dest='coin'
        , action = 'store_false'
        , help = '''If there is no coin in the movie, give this option to skip scaling the data to millimeters.''')
    p.add_argument \
        ( '-wh'
        , '--window-height'
        , metavar = 'px'
        , type = int
        , help = '''Resize the window to a 4:3 landscape with this height. Default behavior is operating system dependent (Linux fits the screen, OSX doesn't).''')

    # main
    exit(main(p.parse_args()))

# eof
