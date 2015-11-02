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
import functools

import cv2
import numpy

import lib.iterutils as iterutils
import lib.cviter as cviter
import lib.circles as circles
import lib.blob_params as blob_params

euclidean_dist = lambda a, b: numpy.sqrt(((a - b) ** 2).sum())
manhattan_dist = lambda a, b: (abs(a - b)).sum()
in_circle = lambda x_y_r, pt, dist=euclidean_dist: dist(x_y_r[:2], pt) <= x_y_r[2]

CV2KeyPoint = cv2.KeyPoint().__class__
BlobPoint = collections.namedtuple('BlobPoint', 'pt keypoint frameinfo')
FlowPoint = collections.namedtuple('FlowPoint', 'pt status error frameinfo')

def new_path(fi, kp):
    assert type(fi).__name__ == 'FrameInfo'
    assert isinstance(kp, CV2KeyPoint)
    return [BlobPoint(pt=numpy.array(kp.pt), keypoint=kp, frameinfo=fi)]

def path_is_active(path):
    # paths should only include blobpoints or status=1 flowpoints
    assert isinstance(path[0], (BlobPoint, FlowPoint))
    latest = path[-1]
    if isinstance(latest, FlowPoint):
        return latest.status == 1
    else:
        return True

def path_loc(path):
    assert isinstance(path[0], (BlobPoint, FlowPoint))
    assert path_is_active(path)
    loc = path[-1].pt
    assert len(loc) == 2
    return loc

def path_dist(path, dist=euclidean_dist):
    # TODO: decide whether we should be including _all_ consecutive displacements
    consec_disps = [dist(t0.pt, t1.pt) for t0, t1 in iterutils.slidingWindow(2, path)]
    return numpy.array(consec_disps).sum()

def path_time_bounds(path):
    begin = path[0].frameinfo.msec
    end = path[-1].frameinfo.msec
    assert begin <= end
    return begin, end

def path_elapsed_time(path):
    begin, end = path_time_bounds(path)
    return end - begin

def flow_path(fi, path, pt_status_err, max_err=100):
    '''return a new Path with the head flowed to the point indicated if ...'''
    pt, status, err = pt_status_err
    assert type(fi).__name__ == 'FrameInfo'
    assert pt.shape == (2,)
    assert status in {0, 1}
    assert isinstance(err, numpy.floating)
    m_disp = manhattan_dist(path[-1].pt, pt_status_err[0])
    return (path[:] + [FlowPoint(pt=pt, status=status, error=err, frameinfo=fi)]) \
            if status == 1 and err < max_err and m_disp < max_err else \
            path

def anchor_path(path, fi, kp):
    '''return a new path with the point at the end'''
    assert type(fi).__name__ == 'FrameInfo'
    assert isinstance(kp, CV2KeyPoint)
    return path[:] + [BlobPoint(pt=numpy.array(kp.pt), keypoint=kp, frameinfo=fi)]

def new_path_group(detect, ti):
    (m0, _), (bim0, _), (_, _) = ti
    return [new_path(m0, kp) for kp in detect(bim0)]

def flow_path_group(pg, ti):
    '''find updated locations for each path head by optical flow
       return
        [(ndarray<2>, 0|1, float)] # for each path, indicate flow-loc, status, and error
    '''
    (_, _), (_, _), (fim0, fim1) = ti
    fs1 = cv2.calcOpticalFlowPyrLK(fim0, fim1,
            numpy.array([path_loc(path) for path in pg], numpy.float32))
    return [(point, status, error)
            for point, (status,), (error,) in itertools.izip(*fs1)]

def anchor_path_group(pg, detect, ti, match_dist=100):
    '''match newly detected blobs against path heads (closer than match_dist)
       return
        ( [CV2KeyPoint|None] # for each path, indicating the best blob or no match
        , [CV2KeyPoint] # for each unmatched blob
        )
    '''
    (_, _), (_, bim1), (_, _) = ti

    # detect new blobs
    bs1 = detect(bim1)

    # blob locations as a matrix
    blob_loc = numpy.empty((len(bs1), 2))
    for B, blob in enumerate(bs1):
        blob_loc[B,:] = blob.pt

    # calculate a table of displacements
    displacement = numpy.empty((len(pg), len(bs1)))
    for P, path in enumerate(pg):
        head = numpy.array(path_loc(path))
        displacement[P,:] = abs(blob_loc - head).sum(axis=1)

    # generate preferences of paths for blobs
    paths_prefd = {}
    for P, path in enumerate(pg):
        # rank all blobs by proximity; filter those off by match_dist
        ranking = displacement[P,:].argsort()
        match_count = (displacement[P,ranking] < match_dist).sum()
        if match_count:
            paths_prefd[P] = collections.deque(
                    itertools.islice(ranking, match_count))

    # gale-shapley
    path_match = {} # {P: CV2KeyPoint}
    blob_match = {} # {B: P}
    def match(P, B):
        # remove an existing match, if any
        if blob_match.get(B) is not None:
            path_match.pop(blob_match.get(B))
        # set up the new match
        path_match[P] = bs1[B]
        blob_match[B] = P

    # continue until all paths with preferences are matched (or no more paths have preferences)
    gen_unmatched = lambda: paths_prefd.viewkeys() - path_match.viewkeys()
    unmatched = gen_unmatched()
    while unmatched:
        for P in unmatched:
            # get an ask; remove this path if it has no more
            B = paths_prefd[P].popleft()
            if not paths_prefd[P]:
                paths_prefd.pop(P)
            # match if the blob is unmatched or this path is a better match
            if B in blob_match:
                if displacement[P,B] < displacement[blob_match[B],B]:
                    match(P, B)
            else:
                match(P, B)
        unmatched = gen_unmatched()

    return [path_match.get(P) for P, path in enumerate(pg)] \
         , [blob for B, blob in enumerate(bs1) if B not in blob_match]

def annot(im, path):
    annot_point(im, path[-1])

def annot_hist(im, path, annotate_point=True, **kwargs):
    if annotate_point:
        annot_point(im, path[-1])
    map(functools.partial(annot_segment, im, **kwargs),
            iterutils.slidingWindow(2, path))

def annot_point(im, point):
    assert isinstance(point, (BlobPoint, FlowPoint))
    if isinstance(point, BlobPoint):
        cv2.drawKeypoints(im, [point.keypoint], im, (25,125,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    else:
        cv2.circle(im, tuple(point.pt), point.error, (50,50,255))

def annot_segment(im, points, color=None, thickness=None):
    if color is None:
        p0, p1 = points
        color = \
            { (BlobPoint, BlobPoint): (0, 0, 0) # black
            , (FlowPoint, FlowPoint): (50,50,255) # deep red
            , (BlobPoint, FlowPoint): (0,255,255) # light yellow
            , (FlowPoint, BlobPoint): (25,125,0) # deep green
            }[type(p0), type(p1)]
    t = 1 if thickness is None else thickness
    cv2.line(im, tuple(map(int, p0.pt)), tuple(map(int, p1.pt)), color, thickness=t)

TrackState = collections.namedtuple('TrackState', 'paths debug')

def gen_flagger(half_petri_x_y_r):
    # flaggers :: {str: (Path -> bool)}
    flaggers = \
        { 'path is comprised of less than 10% BlobPoint'
        : lambda path: len(filter(lambda p: isinstance(p, BlobPoint), path)) / len(path) < 0.1
        , 'path is completely outside of the inner half-radius of the petri dish'
        # TODO: change to manhattan dist
        : lambda path: all(not in_circle(half_petri_x_y_r, p.pt) for p in path)
        }
    # flag :: Path -> Maybe<str> (either a str or None)
    def flagger(path):
        for s,f in flaggers.items():
            if f(path):
                return s
    return flagger

# TODO: wrap CV2KeyPoint & associated FrameInfo in a BlobPoint where detection occurs
# TODO: wrap (pt, status, err) & associated FrameInfo in a FlowPoint where optical-flow occurs

def trackBlobs \
        ( detector
        , half_petri
        , flagger
        , frameinfos
        , debug=None
        , anchor_match_dist=100
        , max_flow_err=100
        , blur_size=5
        , min_filterable_length=10
        ):
    '''iter<ndarray<y,x,3>>[, str] -> ...
    '''
    otf_flagger = lambda path: None if len(path) < min_filterable_length else flagger(path)
    ns = None

    # fan out input streams

    frameinfosA, \
    frameinfosB = itertools.tee(frameinfos)

    frameinfosAframesA, \
    frameinfosAframesB = itertools.tee(itertools.imap(lambda fi: fi.image, frameinfosA))

    # prep input streams

    # blob input can be anything
    blobInput = cviter.buffering \
        ( 2
        , cviter.lift \
            ( lambda fr, out: cv2.blur(fr, (blur_size, blur_size), out)
            , frameinfosAframesA
            )
        )

    # flow input should be 8-bit
    flowInput = cviter.buffering \
        ( 2
        , cviter.gray(frameinfosAframesB)
        )

    # meta input can't use the image (we don't buffer it)
    metaInput = iterutils.slidingWindow \
        ( 2
        , itertools.imap \
            ( lambda fi: fi._replace(image=None)
            , frameinfosB
            )
        )

    # loop on joined input streams
    for ti in itertools.izip(metaInput, blobInput, flowInput):
        (m0, m1), (bim0, bim1), (fim0, fim1) = ti

        # allocate
        if ns is None:
            ns = TrackState \
                ( paths = new_path_group(detector.detect, ti)
                , debug = debug and \
                    ( numpy.empty(bim1.shape[:2] + (3,), bim1.dtype)
                    , numpy.empty(bim1.shape[:2] + (3,), bim1.dtype)
                    )
                )

        # flow (update path heads according to how pixels have moved)
        ns = ns._replace(paths = map \
            ( functools.partial(flow_path, m1, max_err=max_flow_err)
            , ns.paths, flow_path_group(ns.paths, ti)
            ))

        # anchor (redetect blobs and match them against paths)
        anchors, blobs = anchor_path_group(ns.paths, detector.detect, ti,
                match_dist=anchor_match_dist)
        assert len(anchors) == len(ns.paths)

        # paths get a "flow" every frame (where status is 1 and both error & displacement are LT max_flow_error)
        # paths get a "blob" in addition to a flow (when a blob is closer than anchor_match_dist)
        ns = ns._replace(paths \
            = [anchor_path(p, m1, a) if a else p
                for p, a in zip(ns.paths, anchors)]
            + [new_path(m1, b) for b in blobs]
            )

        # filter paths
        ns = ns._replace(paths = iterutils.remove(otf_flagger, ns.paths))

        if debug:
            debugCur, debugHist = ns.debug
            # annotate current
            numpy.copyto(debugCur, bim1)
            cv2.circle(debugCur, tuple(half_petri[:2]), half_petri[2], (128,255,255), 1)
            [annot(debugCur, p) for p in ns.paths]
            # annotate history
            numpy.copyto(debugHist, bim1)
            cv2.circle(debugHist, tuple(half_petri[:2]), half_petri[2], (128,255,255), 1)
            [annot_hist(debugHist, p) for p in ns.paths]
            # annotate differences in frames
            bimdt = cv2.absdiff(bim0, bim1)
            fimdt = cv2.absdiff(fim0, fim1)[...,None] # add extra dim for liken'd debug window
            cviter._debugWindow(debug, trackBlobs.func_name, [bimdt, fimdt, debugCur, debugHist])
        yield ns.paths

# eof
