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
import contextlib

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
                assert self.__video_capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, key), 'able to set capture frame position'
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
            raise TypeError('cannot index {.__name__} by {.__name__}: {}'.format(type(self), type(key), key))

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

    # FIXME: there seems to be a bug where this isn't consistent accross linux/mac
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

    # FIXME: there seems to be a bug where this isn't consistent accross linux/mac
    @staticmethod
    def _read_frame_info(video_capture):
        '''cv2.VideoCapture -> FrameInfo/None

           Extract info about a frame and advance the given VideoCapture object.
           Otherwise, return None.
        '''
        # fetch information about the currently cue'd frame
        fi = Capture.FrameInfo \
            ( index = int(video_capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
            , msec = video_capture.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
            , image = None
            )
        # read the currently cue'd frame AND cue up the following frame
        ret, frame = video_capture.read()
        assert (ret and frame is not None) or (not ret and frame is None), 'ret and frame must agree'
        if ret:
            return fi._replace(image=frame)

class WindowMaker(object):

    def __init__(self, name, flags=None, width_height=None):
        '''str[, int][, (int, int)] -> WindowMaker

           Create a WindowMaker which creates windows with the given flags and
           (width, height).
        '''
        self.__name = name
        self.__flags = flags
        if width_height is not None:
            assert len(width_height) == 2
        self.__w_h = width_height

    def __enter__(self):
        cv2.namedWindow(self.__name) if self.__flags is None else \
        cv2.namedWindow(self.__name, flags=self.__flags)
        self.__w_h and cv2.resizeWindow(self.__name, *self.__w_h)
        return WindowMaker._OpenWindow(self.__name, width_height=self.__w_h)

    def __exit__(self, exc_type, exc_value, traceback):
        cv2.destroyWindow(self.__name)

    class _OpenWindow(object):

        def __init__(self, name, width_height=None):
            self.__name = name
            self.__w_h = width_height
            self.__fb = None
            self.__sf = None

        # for sized windows
        def __init2__(self, ims):
            assert all(im.dtype == ims[0].dtype for im in ims), 'all images must have the same dtype'
            w, h = self.__w_h
            self.__fb = numpy.empty([h, w, 3], dtype=ims[0].dtype)
            self.__sf = self._scale_factor(self._hstack_width_height(ims), (w, h))

        # for sized windows
        def _hstack_width_height(self, ims):
            # TODO: document choice of orientation
            return \
                ( sum(im.shape[1] for im in ims)
                , max(im.shape[0] for im in ims)
                )

        # for sized windows
        def _scale_factor(self, src_size, dst_size):
            src_w, src_h = src_size
            dst_w, dst_h = dst_size
            # ar = w/h
            ## ar(landscape) > 1
            ## ar(portrait) < 1
            src_ar = src_w / src_h
            dst_ar = dst_w / dst_h
            return (dst_h / src_h) if dst_ar > src_ar else \
                   (dst_w / src_w)

        # for sized windows
        def _fb_aoi(self, src_size):
            # TODO: update to facilitate looping over ims to resize them onto fb one at a time
            # TODO: pull out implicit args, such as fb left side which is currently 0
            src_w, src_h = src_size
            return self.__fb \
                [ 0 : round(src_h * self.__sf)
                , 0 : round(src_w * self.__sf)
                ]

        def ims_show(self, ims_, ms=1):
            ims = [ims_] if type(ims_) == numpy.ndarray else ims_

            if self.__w_h is None:
                imshowSafe(self.__name, ims)
            else:
                (self.__fb is None) and self.__init2__(ims)

                # clear fb
                self.__fb.fill(0)

                # TODO: resize images onto fb-aoi one at a time (no allocations)
                src = numpy.hstack(liken(ims))
                dst = self._fb_aoi(self._hstack_width_height(ims))
                cv2.resize \
                    ( src
                    , dsize = tuple(reversed(dst.shape[:2]))
                    , dst = dst
                    , interpolation = cv2.INTER_AREA
                    )

                # blit fb
                cv2.imshow(self.__name, self.__fb)

            # TODO: bring the loop that users of ims_show write into this obj somehow
            return cv2.waitKey(ms)

        def on_mouse(self, callback):
            if callback is None:
                on_mouse_wrapper = lambda *_: None
            else:
                @functools.wraps(callback)
                def on_mouse_wrapper(event, x, y, flags, param):
                    scale = lambda n: int(n / (self.__sf or 1))
                    return callback(event, scale(x), scale(y), flags, param)
            cv2.setMouseCallback(self.__name, on_mouse_wrapper)

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
    assert all(map(lambda im: im.ndim in {2,3}, ims0)), 'liken only supports arrays with ndim==2 or ndim==3, got {}'.format(map(lambda im: im.shape, ims0))
    assert all(map(lambda im: im.dtype == ims0[0].dtype, ims0)), 'liken currently only supports groups of arrays with the same dtype, got {}'.format(map(lambda im: im.dtype, ims0))
    # [overwrite ims0] force single-channel images to have ndim==3
    ims0 = [im[..., None] if im.ndim == 2 else im for im in ims0]
    # find the maximum width (x), height (y) and depth(d) among the images
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
