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

PxStats = collections.namedtuple('PxStats', 'min max median mean std')

def pxStats(im):
    '''ndarray<y,x,d> -> PxStats<ndarray<d>>

       Calculate component-wise statistics over all pixels in the image.
       std ** 2 == var

       Here's an example using a BGRA image:

       >>> from numpy import *
       >>> st = pxStats(array \
               ([[[  0, 255, 102, 255],  \
                  [  0, 255, 102,  85],  \
                  [  0,   0,   0,   0]], \
                 [[  0, 255, 102, 255],  \
                  [ 41, 153, 163, 142],  \
                  [102,   0, 255,  85]], \
                 [[  0, 255, 102, 255],  \
                  [ 68,  85, 204, 255],  \
                  [102,   0, 255, 255]]], dtype=uint8))
       >>> st.min
       array([0, 0, 0, 0], dtype=uint8)
       >>> st.max
       array([102, 255, 255, 255], dtype=uint8)
       >>> st.median
       array([   0.,  153.,  102.,  255.])
       >>> st.mean
       array([  34.77777778,  139.77777778,  142.77777778,  176.33333333])
       >>> st.std
       array([  42.46247436,  112.98650635,   79.14933533,   94.22078091])
    '''
    w, h, d = im.shape
    px = im.reshape([w * h, d])
    return PxStats \
        ( min = px.min(axis=0)
        , max = px.max(axis=0)
        , median = numpy.median(px, axis=0)
        , mean = px.mean(axis=0)
        , std = px.std(axis=0)
        )

class Capture(collections.namedtuple('Capture', 'source capture')):
    '''An iterator wrapper for the OpenCV capture type.'''
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

def alphaBlend(fg, bg, dst=None):
    '''numpy.ndarray<y,x,4>, numpy.ndarray<y,x,4> -> numpy.ndarray<y,x,4>

       Alpha blend BGRA `fg` onto `bg` in a newly allocated array.
    '''
    if dst is None:
        dst = numpy.empty_like(bg)

    scale = 1.0 / 255
    fgBGR = fg[..., :3].astype(numpy.float32) * scale
    bgBGR = bg[..., :3].astype(numpy.float32) * scale
    fgA = (fg[..., 3].astype(numpy.float32) * scale)[..., None]
    bgA = (bg[..., 3].astype(numpy.float32) * scale)[..., None]

    outA = fgA + bgA * (1 - fgA)
    outBGR = ( fgBGR * fgA
             + bgBGR * bgA * (1 - fgA)
             ) / outA

    dst[..., :3] = (outBGR * 255).round()
    dst[..., 3] = (outA * 255).round()[..., 0]
    return dst

def gray2color(src, dst=None):
    return cv2.merge([src, src, src], dst)

def explain(msg, im, pr=False):
    '''Gush about an ndarray.'''
    print(im.ndim, im.shape, im.dtype, im.size, msg)
    if pr:
        print(im)

def liken(ims0):
    ''' make dissimilar images similar enough to display with imshow '''
    assert all(map(lambda im: im.ndim == 3, ims0)), 'liken currently only supports arrays with ndim==3, got {}'.format(map(lambda im: im.shape, ims0))
    assert all(map(lambda im: im.dtype == ims0[0].dtype, ims0)), 'liken currently only supports groups of arrays with the same dtype, got {}'.format(map(lambda im: im.dtype, ims0))
    tx, ty, td = reduce(lambda whd, im: map(max, zip(whd, im.shape)), ims0, [0,0,0])
    template = numpy.zeros([tx, ty, td], ims0[0].dtype)
    if td > 3:
        template[..., 3] = 255
    ims1 = []
    for im in ims0:
        imx, imy, imd = im.shape
        if imd == td:
            ims1.append(im)
        else:
            t = template.copy()
            print('Warning: liken allocated {},{} to accomodate a {},{}'.format(t.shape,t.dtype, im.shape,im.dtype))
            t[:imx,:imy,:max(imd, td)] = im[...]
            ims1.append(t)
    else:
        return ims1

def imshowSafe(window, ims):
    ''' display images with imshow '''
    if type(ims) == numpy.ndarray:
        ims = [ims]
    try:
        cv2.imshow(window, numpy.hstack(liken(ims)))
    except ValueError as e:
        map(functools.partial(explain, ''), ims)
        print('Error:', e)
        exit(1)

def circle(diameter, color=255):
    ''' create a new image containing a centered circle '''
    arr = numpy.zeros((diameter, diameter), numpy.uint8)
    m = diameter // 2
    cv2.circle(arr, (m, m), m, color, -1)
    return arr

# eof
