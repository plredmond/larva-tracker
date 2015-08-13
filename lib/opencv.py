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

# eof
