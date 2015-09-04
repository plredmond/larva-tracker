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
import lib.blob_params as blob_params

Path = collections.namedtuple('Path', 'hist')

KeyPoint = cv2.KeyPoint().__class__
FlowPoint = collections.namedtuple('FlowPoint', 'pt status error')

def new_path(point):
    assert isinstance(point, KeyPoint)
    return Path(hist=[point])

def path_is_active(path):
    assert isinstance(path, Path)
    latest = path.hist[-1]
    if isinstance(latest, FlowPoint):
        return latest.status == 1
    else:
        return True

def path_loc(path):
    assert isinstance(path, Path)
    assert path_is_active(path)
    loc = path.hist[-1].pt
    assert len(loc) == 2
    return loc

# FIXME: pay attention to error
def flow_path(path, pt_status_err, max_err=100):
    '''return a new Path with the head flowed to the point indicated if ...'''
    pt, status, err = pt_status_err
    assert pt.shape == (2,)
    assert status in {0, 1}
    assert isinstance(err, numpy.floating)
    if status == 1 and err < max_err:
        return Path(hist=path.hist[:] + [FlowPoint(pt=pt, status=status, error=err)])
    else:
        return path

def anchor_path(path, point):
    '''return a new path with the point at the end'''
    assert isinstance(point, KeyPoint)
    return Path(hist=path.hist[:] + [point])

def new_path_group(detect, ims):
    (bim0, _), (_, _) = ims
    return [new_path(kp) for kp in detect(bim0)]

def flow_path_group(pg, ims):
    '''find updated locations for each path head by optical flow
       return
        [(ndarray<2>, 0|1, float)] # for each path, indicate flow-loc, status, and error
    '''
    (_, _), (fim0, fim1) = ims
    fs1 = cv2.calcOpticalFlowPyrLK(fim0, fim1,
            numpy.array([path_loc(path) for path in pg], numpy.float32))
    return [(point, status, error)
            for point, (status,), (error,) in itertools.izip(*fs1)]

def anchor_path_group(pg, detect, ims, match_dist=100):
    '''match newly detected blobs against path heads (closer than match_dist)
       return
        ( [KeyPoint|None] # for each path, indicating the best blob or no match
        , [KeyPoint] # for each unmatched blob
        )
    '''
    (_, bim1), (_, _) = ims

    # detect new blobs
    bs1 = detect(bim1)

    # blob locations as a matrix
    blob_loc = numpy.empty((len(bs1), 2))
    for B, blob in enumerate(bs1):
        blob_loc[B,:] = blob.pt
    del B, blob

    # calculate a table of displacements
    displacement = numpy.empty((len(pg), len(bs1)))
    for P, path in enumerate(pg):
        head = numpy.array(path_loc(path))
        displacement[P,:] = abs(blob_loc - head).sum(axis=1)
    del P, path, head
    del blob_loc

    # generate preferences of paths for blobs
    paths_prefd = {}
    for P, path in enumerate(pg):
        # rank all blobs by proximity; filter those off by match_dist
        ranking = displacement[P,:].argsort()
        match_count = (displacement[P,ranking] < match_dist).sum()
        if match_count:
            paths_prefd[P] = collections.deque(
                    itertools.islice(ranking, match_count))
    del P, path, ranking, match_count

    # gale-shapley
    path_match = {} # {P: KeyPoint}
    blob_match = {} # {B: P}
    def match(P, B):
        # remove an existing match, if any
        if blob_match.get(B) is not None:
            path_match.pop(blob_match.get(B))
        # set up the new match
        path_match[P] = bs1[B]
        blob_match[B] = P

    # continue until all paths with preferences are matched (or no more paths have preferences)
    pd = lambda d: '\n' + '\n'.join('  %s' % repr(x) for x in sorted(d.items()))
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
    del match, pd, gen_unmatched, unmatched, P, B

    return [path_match.get(P) for P, path in enumerate(pg)] \
         , [blob for B, blob in enumerate(bs1) if B not in blob_match]

def annot(im, path_group):
    map(functools.partial(annot_point, im), [path.hist[-1] for path in path_group])

def annot_hist(im, path_group):
    for path in path_group:
        annot_point(im, path.hist[-1])
        map(functools.partial(annot_segment, im),
                iterutils.slidingWindow(2, path.hist))

def annot_point(im, point):
    if isinstance(point, KeyPoint):
        cv2.drawKeypoints(im, [point], im, (25,125,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    else:
        cv2.circle(im, tuple(point.pt), point.error, (50,50,255))

def annot_segment(im, points):
    p0, p1 = points
    color = \
        { (KeyPoint, KeyPoint): (0, 0, 0) # black
        , (FlowPoint, FlowPoint): (50,50,255) # deep red
        , (KeyPoint, FlowPoint): (0,255,255) # light yellow
        , (FlowPoint, KeyPoint): (25,125,0) # deep green
        }[type(p0), type(p1)]
    cv2.line(im, tuple(map(int, p0.pt)), tuple(map(int, p1.pt)), color)

TrackState = collections.namedtuple('TrackState', 'paths annotCur annotHist')

# rules return true to identify invalid paths
on_the_fly_filters = \
    [ lambda path: len(path.hist) >= 10 and \
        len(filter(lambda p: isinstance(p, KeyPoint), path.hist)) / len(path.hist) < 0.1
    ]

def filter_on_the_fly(path_group):
    return filter(lambda path: not any(f(path) for f in on_the_fly_filters), path_group)


def trackBlobs \
        ( detector
        , frames
        , debug=None
        , anchor_match_dist=100
        , max_flow_err=100
        ):
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
    for ims in itertools.izip(blobInput, flowInput):
        (bim0, bim1), (fim0, fim1) = ims
        # allocate
        if ns is None:
            ns = TrackState \
                ( paths = new_path_group(detector.detect, ims)
                , annotCur = numpy.empty(bim1.shape[:2] + (3,), bim1.dtype)
                , annotHist = numpy.empty(bim1.shape[:2] + (3,), bim1.dtype)
                )
        # flow (update path heads according to how pixels have moved)
        ns = ns._replace(paths = map \
            ( functools.partial(flow_path, max_err=max_flow_err)
            , ns.paths, flow_path_group(ns.paths, ims)
            ))
        # anchor (redetect blobs and match them against paths)
        anchors, blobs = anchor_path_group(ns.paths, detector.detect, ims,
                match_dist=anchor_match_dist)
        assert len(anchors) == len(ns.paths)
        ns = ns._replace(paths \
            = [anchor_path(p, a) if a else p
                for p, a in zip(ns.paths, anchors)]
            + [new_path(b) for b in blobs]
            )
        del anchors, blobs

        # paths get a "flow" every frame (where status is 1 and error is less than max_flow_err)
        # paths get a "blob" in addition to a flow (when a blob is closer than anchor_match_dist)

        # then we remove paths which don't fit certain specs
        ns = ns._replace(paths = filter_on_the_fly(ns.paths))

        # annotate current
        numpy.copyto(ns.annotCur, bim1)
        annot(ns.annotCur, ns.paths)

        # annotate history
        numpy.copyto(ns.annotHist, bim1)
        annot_hist(ns.annotHist, ns.paths)

        if debug:
            bimdt = cv2.absdiff(bim0, bim1) # [...,None]
            fimdt = cv2.absdiff(fim0, fim1)[...,None]
            cviter._debugWindow(debug, trackBlobs.func_name, [bimdt, fimdt, ns.annotCur, ns.annotHist])
        yield (ns.annotCur, ns.annotHist, ns.paths)

# eof
