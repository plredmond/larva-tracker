from __future__ import \
    ( nested_scopes
    , generators
    , division
    , absolute_import
    , with_statement
    , print_function
    , unicode_literals
    )

import functools

import numpy
import cv2


class Point2D(object):
    '''Since numpy talks about (row, column) and cv2 talks about (x, y).'''
    __slots__ = '__arr'
    def __init__(self, **kwargs):
        self.__arr = numpy.array \
            ( { frozenset({'x', 'y'}): lambda d: (d['y'], d['x'])
              , frozenset({'r', 'c'}): lambda d: (d['r'], d['c'])
              }[frozenset(kwargs.viewkeys())](kwargs)
            )
    @property
    def row(self):
        return self.__arr[0]
    @property
    def col(self):
        return self.__arr[1]
    @property
    def rc(self):
        return tuple(self.__arr)
    @property
    def rc_arr(self):
        return numpy.array(self.rc)
    @property
    def xy(self):
        return tuple(reversed(self.__arr))
    @property
    def xy_arr(self):
        return numpy.array(self.xy)
    def __len__(_):
        return 2
    def __cmp__(*_):
        raise NotImplementedError('cannot compare')
    def __eq__(self, other):
        return (self.__arr == other.rc_arr).all()
    def __repr__(self):
        return '{}(r={},c={})'.format(type(self).__name__, repr(self.row), repr(self.col))
    def __str__(self):
        return '<row={:g},col={:g}>'.format(*self.__arr)


class Box(object):
    __slots__ = ('__arr')
    def __init__(self, a, b):
        '''Point2D, Point2D -> Box'''
        if (a.rc_arr <= b.rc_arr).all():
            self.__arr = numpy.vstack((a.rc_arr, b.rc_arr))
        else:
            raise ValueError('points must represent a box')
    @property
    def pt0(self):
        [row, col], _  = self.__arr
        return Point2D(r=row, c=col)
    @property
    def pt1(self):
        _, [row, col]  = self.__arr
        return Point2D(r=row, c=col)
    @property
    def pt_rc_arr(self):
        return self.__arr.copy()
    @property
    def slices(self):
        return tuple(map(functools.partial(apply, slice), self.__arr.transpose()))
    def rectangle(self, img, color, **kwargs):
        '''numpy.ndarray, tup<num>[, kwargs of cv2.rectangle] -> None'''
        pt0, pt1 = numpy.fliplr(self.__arr)
        return cv2.rectangle(img, tuple(pt0), tuple(pt1), color, **kwargs)
    def crop(self, img):
        '''numpy.ndarray -> numpy.ndarray'''
        return img[self.slices]
    def expand(self, n):
        '''num -> Box'''
        pts = self.pt_rc_arr
        pts[0] -= n
        pts[1] += n
        return type(self)(pts)
    def __repr__(self):
        [r0, c0], [r1, c1] = self.__arr
        return '{}({}, {})'.format(
                type(self).__name__,
                repr(Point2D(r=r0,c=c0)),
                repr(Point2D(r=r1, c=c1)))
    @classmethod
    def from_rc_arr(cls, arr):
        [r0, c0], [r1, c1] = arr
        return Box(Point2D(c=c0, r=r0), Point2D(c=c1, r=r1))

# eof
