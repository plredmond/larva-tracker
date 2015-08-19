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

import numpy
import cv2

import lib.iterutils as iterutils
import lib.opencv as opencv

def _debugWindow(name, itername, arrs, t=1):
    '''str, [ndarray] -> None

       Display a debug window containing the images left-to-right for `t` ms.
    '''
    if name:
        name = 'debug|{}|{}'.format(name, itername)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        opencv.imshowSafe(name, arrs)
        cv2.waitKey(t)

def displaySink(name, stream, t=1, ending=False, quit=27):
    '''str, iter<[ndarray]>[, int] -> None

       Consume sequences of images by displaying them left-to-right in a window for `t` ms.
       If `ending` then display the last image indefinitely.
       In either case, return upon receipt of `quit` keycode (default is ESC for me).

       TODO: maybe convert this to an iter that simply yields (arrs, show) or (arrs, key)
    '''
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    for arrs in stream:
        def show(n):
            opencv.imshowSafe(name, arrs)
            return cv2.waitKey(n)
        if show(t) == quit:
            return
    while ending:
        if show(0) == quit:
            return

# an iter asserts required properties about its inputs
# an iter allocates workspace only once at the beginning
# an iter performs work by copying from inputs to workspace; it never modifies input
# an iter yields work by providing a pointer to a portion of its workspace
# -> an iter must not hold references to previous inputs for use in subsequent iterations

def gray(frames, debug=None):
    '''iter<ndarray>[, str] -> iter<ndarray>

       Iterate over grayscale versions of BGR/BGRA images.
    '''
    gr = None
    for fr in frames:
        assert fr.ndim == 3 and fr.shape[2] in {3, 4}
        if gr is None:
            gr = numpy.empty(fr.shape[:2] + (1,), fr.dtype)
        cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY if fr.shape[2] == 3 else cv2.COLOR_BGRA2GRAY, gr)
        _debugWindow(debug, gray.func_name, [fr, gr])
        yield gr

def allocating(frames, debug=None):
    '''iter<ndarray>[, str] -> iter<ndarray>

       Iterate over copies of upstream frames.
       This allows a downstream iter to reuse previous values of inputs.
       This results in massive allocation and should be avoided.
    '''
    kinds = collections.defaultdict(int)
    for fr in frames:
        kinds[fr.shape,fr.dtype,fr.size] += 1
        y = fr.copy()
        _debugWindow(debug, allocating.func_name, y)
        yield y
    print('Warning: made allocations', ''.join('\n- {1} x {0[2]} bytes for {0[0]} {0[1]}'.format(*i) for i in kinds.items()))

def buffering(size, frames, debug=None):
    '''int, iter<ndarray>[, str] -> iter<ndarray>

       Iterate over every `size` length subsequence of upstream frames.
       This allows a downstream iter to reuse previous values of inputs.
    '''
    buff = None
    seen = 0
    for fr in frames:
        # allocate all of the buffer at once
        if buff is None:
            buff = collections.deque([numpy.empty_like(fr) for _ in range(size)])
        # assert that the current thing is compatible with the buffer
        assert fr.size == buff[0].size and fr.dtype == buff[0].dtype
        # process the current thing
        numpy.copyto(buff[0], fr)
        buff.append(buff.popleft())
        seen += 1
        if seen >= size:
            y = list(buff)
            _debugWindow(debug, buffering.func_name, y)
            yield y

def motion(frames, era=None, debug=None):
    '''iter<ndarray>[, str][, str] -> iter<ndarray>

       Iterate over the absdiff of subsequent pairs of frames.
    '''
    f = { 'span': lambda t0, t1, dt: cv2.absdiff(t0, t1, dt)
        , 'past': lambda t0, t1, dt: cv2.subtract(t0, t1, dt)
        , 'future': lambda t0, t1, dt: cv2.subtract(t1, t0, dt)
        }[era or 'span']
    dt = None
    for (t0, t1) in buffering(2, frames, debug):
        assert id(t0) != id(t1)
        if dt is None:
            dt = numpy.empty_like(t0)
        f(t0, t1, dt)
        _debugWindow(debug, motion.func_name, [t0, t1, dt])
        yield dt

def fork(ways, frames, debug=None):
    '''int, iter<ndarray>[, str] -> [iter<ndarray>, ...]

       Duplicate `frames` by allocating a copy of every frame and using `itertools.tee`.
       Return `ways` new iterators.
    '''
    return itertools.tee(allocating(frames, debug), ways)

def alphaBlended(pairs, debug=None):
    '''int, iter<(ndarray<x,y,4>, ndarray<x,y,4>)> -> iter<ndarray<x,y,4>>

       Iterate the result of alpha blending incoming pairs of (foreground, background).
    '''
    dst = None
    for (fg, bg) in pairs:
        if dst is None:
            dst = numpy.empty_like(bg)
        assert 3 == fg.ndim == bg.ndim == dst.ndim, 'alphaBlend requires color images (3 dimensions)'
        assert 4 == fg.shape[2] == bg.shape[2] == dst.shape[2], 'alphaBlend requires images with an alpha layer (depth of 4 values)'
        assert fg.shape == bg.shape == dst.shape, 'alphaBlend requires images of the same dimensions'
        opencv.alphaBlend(fg, bg, dst)
        _debugWindow(debug, alphaBlended.func_name, [fg, bg, dst])
        yield dst

def cvtColor(code, dstCn, frames, debug=None):
    dst = None
    for fr in frames:
        if dst is None:
            dst = numpy.empty(fr.shape[:2] + (dstCn,), dtype=fr.dtype)
        cv2.cvtColor(fr, code, dst, dstCn)
        _debugWindow(debug, cvtColor.func_name, [fr, dst])
        yield dst

def applyTo(extract, restore, partialIter, upstream):
    '''(a -> b), (a, c -> d), (iter<b> -> iter<c>), iter<a> -> iter<d>

       Iterate over a modification of `upstream` in which each item has been `extract`ed passed through `partialIter` and `restore`ed.

       >>> import functools, itertools
       >>> list(applyTo(
       ...     lambda (x, y0): y0,
       ...     lambda (x, _), y1: (x, y1),
       ...     functools.partial(itertools.imap, lambda y: y ** 2),
       ...     [('a', 2), ('b', 9)]))
       [('a', 4), ('b', 81)]
    '''
    passthru, toproc = itertools.tee(upstream, 2)
    return itertools.imap(restore, passthru, partialIter(itertools.imap(extract, toproc)))

ffirst  = functools.partial(applyTo, lambda (a,_): a, lambda (_,b),a: (a,b))
ssecond = functools.partial(applyTo, lambda (_,b): b, lambda (a,_),b: (a,b))
fffirst  = functools.partial(applyTo, lambda (a,_,__): a, lambda (_,b,c),a: (a,b,c))
sssecond = functools.partial(applyTo, lambda (_,b,__): b, lambda (a,_,c),b: (a,b,c))
ttthird  = functools.partial(applyTo, lambda (_,b,__): b, lambda (a,_,c),b: (a,b,c))

def lift \
        ( fn
        , upstream
        , egfn = None
        , shapefn = None
        , dtypefn = None
        , allocfn = None
        , debug = None
        ):
    egfn = egfn or (lambda x: x)
    shapefn = shapefn or (lambda x: x.shape)
    dtypefn = dtypefn or (lambda x: x.dtype)
    allocfn = allocfn or (lambda x: numpy.empty(shapefn(x), dtypefn(x)))
    ns = None
    for x in upstream:
        if ns is None:
            ns = allocfn(egfn(x))
        fn(x, ns)
        _debugWindow(debug, lift.func_name, ns)
        yield ns

def fgMask(frames, debug=None):
    model = cv2.BackgroundSubtractorMOG()
    return lift \
        ( lambda fr, mask: model.apply(fr, mask)
        , frames
        , debug=debug
        )

# eof
