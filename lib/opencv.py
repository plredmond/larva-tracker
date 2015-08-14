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
import argparse
import functools

import numpy
import cv2

class Capture(collections.namedtuple('Capture', 'source capture')):
    __slots__ = ()
    @classmethod
    def argtype(cls, source):
        '''str/int -> Capture

           Construct a Capture from the path/device or give an argument type error.
        '''
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            return Capture(source, cap)
        else:
            raise argparse.ArgumentTypeError('source {0}'.format(source))
    def duplicate(self):
        '''-> Capture

           Return a capture of the same path/device, reset to the current frame therin.
        '''
        return self.argtype(self.source)
    def __iter__(self):
        '''-> iter<numpy.ndarray>'''
        return self
    def next(self):
        '''-> numpy.ndarray'''
        ret, frame = self.capture.read()
        assert (ret and frame is not None) or (not ret and frame is None), 'ret and frame must agree'
        if ret:
            return frame
        else:
            self.capture.release()
            raise StopIteration
    def __repr__(self):
        return 'Capture.argtype({})'.format(repr(self.source))

def alphaBlend(src, dst):
    '''numpy.ndarray<x,y,4>, numpy.ndarray<x,y,4> -> numpy.ndarray<x,y,4>

       Alpha blend BGRA `src` onto `dst` in a newly allocated array.
    '''
    srcBGR = src[..., :3].astype(numpy.float32) / 255
    dstBGR = dst[..., :3].astype(numpy.float32) / 255
    srcA = src[..., 3].astype(numpy.float32) / 255
    dstA = dst[..., 3].astype(numpy.float32) / 255
    #
    outA = srcA + dstA * (1 - srcA)
    outBGR = ( srcBGR * srcA[..., None]
             + dstBGR * dstA[..., None] * (1 - srcA[..., None])
             ) / outA[..., None]
    #
    out = numpy.zeros_like(dst)
    out[..., :3] = (outBGR * 255).round()
    out[..., 3] = (outA * 255).round()
    return out

def gray2color(src, dst=None):
    return cv2.merge([src, src, src], dst)

def explain(msg, im, pr=False):
    '''Gush about an ndarray.'''
    print(im.ndim, im.shape, im.dtype, im.size, msg)
    if pr:
        print(im)

def imshowSafe(window, *ims):
    try:
        cv2.imshow(window, numpy.hstack(ims))
    except ValueError as e:
        map(functools.partial(explain, ''), ims)
        print('Error:', e)
        exit(1)

# eof
