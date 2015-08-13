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

# source is a generator
# pipe stages take upstream iter and return downstream iter
# sink is a reducer, eg list() or reduce()

fork = itertools.tee
join = itertools.izip_longest
race = itertools.izip

def slidingWindow(size, upstream):
    '''int iter<a> -> iter<[a]>

       Iterate over every `size` length subsequence of `upstream`.

       >>> list(slidingWindow(1, list("hello")))
       [['h'], ['e'], ['l'], ['l'], ['o']]
       >>> list(slidingWindow(3, list("hello")))
       [['h', 'e', 'l'], ['e', 'l', 'l'], ['l', 'l', 'o']]
       >>> list(slidingWindow(5, list("hello")))
       [['h', 'e', 'l', 'l', 'o']]
       >>> list(slidingWindow(6, list("hello")))
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
