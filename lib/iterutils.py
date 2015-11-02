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

# a source is a generator
# a pipe stage is a generator which takes an upstream iter
# a sink is a reducer, eg list() or reduce()

fork = itertools.tee
join = itertools.izip_longest
race = itertools.izip


keep = filter
def remove(callbackM, sequence):
    '''remove(function or None, sequence) -> list, tuple, or string

       Return those items of sequence for which function(item) is false.  If
       function is None, return the items that are false.  If sequence is a tuple
       or string, return the same type, else return a list.

       >>> odd = lambda n: n % 2
       >>> keep(odd, range(10))
       [1, 3, 5, 7, 9]
       >>> remove(odd, range(10))
       [0, 2, 4, 6, 8]
       >>> keep(None, u'comma,separated,,values,'.split(','))
       [u'comma', u'separated', u'values']
       >>> remove(None, u'comma,separated,,values,'.split(','))
       [u'', u'']
    '''
    f = (lambda x: not x) if callbackM is None else (lambda x: not callbackM(x))
    return filter(lambda x: f(x), sequence)


def slidingWindow(size, upstream):
    '''int iter<a> -> iter<[a]>

       Iterate over every `size` length subsequence of `upstream`.

       >>> list(slidingWindow(1, list(b'hello')))
       [['h'], ['e'], ['l'], ['l'], ['o']]
       >>> list(slidingWindow(3, list(b'hello')))
       [['h', 'e', 'l'], ['e', 'l', 'l'], ['l', 'l', 'o']]
       >>> list(slidingWindow(5, list(b'hello')))
       [['h', 'e', 'l', 'l', 'o']]
       >>> list(slidingWindow(6, list(b'hello')))
       []
    '''
    buff = collections.deque([], size)
    for v in upstream:
        buff.append(v)
        if len(buff) == size:
            yield list(buff)

def trueEvery(n):
    '''n -> iter<bool>
    
       Infinite. Produce True on every `n`th iteration, starting with the first.

       >>> list(itertools.islice(trueEvery(1), 4))
       [True, True, True, True]
       >>> list(itertools.islice(trueEvery(2), 4))
       [True, False, True, False]
       >>> list(itertools.islice(trueEvery(3), 4))
       [True, False, False, True]
    '''
    ct = 0
    while True:
        yield ct == 0
        ct = (ct + 1) % n

# eof
