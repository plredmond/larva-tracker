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

import numpy
import cv2

import lib.cviter as cviter
import lib.trackingiter as triter
import lib.opencv as opencv
import lib.iterutils as iterutils
import lib.funcutils as funcutils

def prepDisplay(stream, debug=None):
    ns = None
    for ((_, _, _, featureHist), featAnnot, pathAnnot) in stream:
        # initialize ns
        if ns is None:
            pass
        # proc
        # yield
        cviter._debugWindow(debug, prepDisplay.func_name, [pathAnnot, featAnnot])
        yield (pathAnnot, featAnnot)

def main(args):
    # print args
    map(lambda s: print('{}: {}'.format(s, getattr(args, s))),
            filter(lambda s: s[0] != '_',
                sorted(dir(args))))

    windowName = '{0} - {1}'.format(path.basename(sys.argv[0]), args.movie.source)

    # TODO: find a way to express fork/tee and join/zip with nesting structure, wrap it all up

    m_dropped = itertools.islice(args.movie, args.drop, None)
    am_dropped, bm_dropped = itertools.tee(m_dropped, 2)

    bm_gray = cviter.gray(bm_dropped)
    bm_motion = cviter.motion(bm_gray)
    bm_tracked = triter.trackCorners(bm_motion,
           redetectInterval=args.redetect if args.redetect != -1 else None)
    abm_tracked, bbm_tracked, cbm_tracked = itertools.tee(bm_tracked, 3)
    # TODO: try different args to goodFeaturesToTrack and calcOpticalFlowPyrLK

    bbm_feats = triter.annotateFeatures(bbm_tracked)
    bbm_feata = cviter.ssecond(functools.partial(cviter.cvtColor, cv2.COLOR_GRAY2BGRA, 4), bbm_feats)
    bbm_annot = cviter.alphaBlended(bbm_feata)

    cbm_paths = triter.annotatePaths(cbm_tracked)
    cbm_fgbg = itertools.imap(lambda fr, (annot, _): (annot, fr), am_dropped, cbm_paths)
    cbm_fgbga = cviter.ssecond(functools.partial(cviter.cvtColor, cv2.COLOR_BGR2BGRA, 4), cbm_fgbg)
    cbm_annot = cviter.alphaBlended(cbm_fgbga)

    m_stream = itertools.izip(abm_tracked, bbm_annot, cbm_annot)
    m_display = prepDisplay(m_stream)
    cviter.displaySink(windowName, m_display, ending=True)

    # - find the blob of moving larva in the center (area of interest)
    #   - show area of interest as darkening mask in lhs
    #     http://docs.opencv.org/master/d0/d86/tutorial_py_image_arithmetics.html#gsc.tab=0
    # - for analysis, ignore stuff outside of AOI
    # - do feature detection on the hsl-tweaked AOI image
    # - increase the diameter of the AOI according to the flow info
    # - find the red penny, measure, erase it

sentinel = \
    { 'disable-redetect': -1
    }

default = \
    { 'redetect': 20
    , 'drop': 0
    }

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('movie',
            type=opencv.Capture.argtype,
            help='''The path to the movie file to perform image tracking on.''')
    p.add_argument('-r', '--redetect',
            default=default['redetect'], type=int, metavar='I',
            help='''Interval number of frames to wait before redoing feature
            selection. (default {deft} frames, use {dis} to disable
            redetection)'''.format(
                deft=default['redetect'], dis=sentinel['disable-redetect']))
    p.add_argument('-d', '--drop',
            default=default['drop'], type=int, metavar='D',
            help='''Number of frames to drop from the beginning of the analysis.
            (default {deft})'''.format(deft=default['drop']))
    main(p.parse_args())

# eof
