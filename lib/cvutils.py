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
import itertools

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


class Capture(object):

    def __init__(self, source):
        '''str/int => Capture

           Construct a Capture object from file-path or device-number.
        '''
        self.__source = source
        self.__video_capture = cv2.VideoCapture(source)
        if not self.__video_capture.isOpened():
            raise ValueError(source)

    def duplicate(self):
        '''-> Capture

           Return a capture of the same originating file-path or device-number at the current frame therin.
        '''
        return type(self)(self.__source)

    def __getitem__(self, key):
        '''int -> FrameInfo : Return a frame from this capture.
           slice -> iter<FrameInfo> : Return a lazy iter over frames from a duplicate of this capture.
        '''
        if isinstance(key, int):
            if 0 <= key < self.frame_count:
                assert self.capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, key), 'able to set capture frame position'
                fi = self._read_frame_info(self.__video_capture)
                assert fi is not None, 'resulting frame position produced data'
                assert fi.index == key, 'resulting frame position is as was expected'
                return fi
            else:
                raise IndexError(key)
        elif isinstance(key, slice):
            # TODO: skip directly to key.start with set(cv2.cv.CV_CAP_PROP_POS_FRAMES, key.start)
            return itertools.islice(iter(self), key.start, key.stop, key.step)
        else:
            raise TypeError('cannot index {} by {}: {}'.format(type(self), type(key), key))

    class _Iter(object):

        '''Hygienic iterator implementation for the Capture class.'''

        def __init__(self, video_capture):
            '''cv2.VideoCapture => CaptureIter'''
            self.__video_capture = video_capture

        def __iter__(self):
            return self

        def next(self):
            '''-> FrameInfo'''
            fi = Capture._read_frame_info(self.__video_capture)
            if fi is None:
                self.__video_capture.release()
                raise StopIteration
            else:
                return fi

    def __iter__(self):
        '''-> iter<FrameInfo>'''
        return Capture._Iter(self.duplicate().__video_capture)

    def __repr__(self):
        return '{.__name__}({})'.format(type(self), repr(self.__source))

    @property
    def source(self):
        '''str'''
        return self.__source

    @property
    def fps(self):
        '''float'''
        return self.__video_capture.get(cv2.cv.CV_CAP_PROP_FPS)

    @property
    def frame_count(self):
        '''int'''
        return int(self.__video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    @property
    def frame_height(self):
        '''int'''
        return int(self.__video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_width(self):
        '''int'''
        return int(self.__video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))

    @classmethod
    def argtype(cls, source):
        '''Factory function for argparse'''
        try:
            return cls(source)
        except ValueError:
            raise argparse.ArgumentTypeError('source {0}'.format(source))

    FrameInfo = collections.namedtuple('FrameInfo', 'index msec image')

    @staticmethod
    def _read_frame_info(video_capture):
        '''cv2.VideoCapture -> FrameInfo/None

           Extract info about a frame and advance the given VideoCapture object.
           Otherwise, return None.
        '''
        ret, frame = video_capture.read()
        assert (ret and frame is not None) or (not ret and frame is None), 'ret and frame must agree'
        if ret:
            return Capture.FrameInfo \
                ( index = int(video_capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
                , msec = video_capture.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
                , image = frame
                )

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
