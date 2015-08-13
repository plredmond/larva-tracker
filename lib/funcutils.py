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

def flip(f0):
    '''(a, b, ... -> c) -> (..., b, a -> c)'''
    @functools.wraps(f0)
    def f1(*args, **kwargs):
        return f0(*args[::-1], **kwargs)
    return f1

def curry(f0):
    '''([a, b, ...] -> c) -> (a, b, ... -> c)'''
    @functools.wraps(f0)
    def f1(*args, **kwargs):
        return f0(args, **kwargs)
    return f1

def uncurry(f0):
    '''(a, b, ... -> c) -> ([a, b, ...] -> c)'''
    @functools.wraps(f0)
    def f1(args, **kwargs):
        return f0(*args, **kwargs)

__ap = lambda datum, fun: fun(datum)
# apply :: (a, b, ... -> c), [a, b, ...] -> c
#   - uncurry the first arg to change its args to a seq
#       :: ([a, b, ...] -> c), [a, b, ...] -> c
#   - simplify all occurances of [a, b, ...] to d
#     (removing all instances of the type variables therin)
#       :: (d -> c), d -> c
#   - flip the argument order
# ap    :: a, (a -> b) -> b

# pass datums through a list of functions expressing a pipeline
# pipeline :: [(a -> b), (b -> c), ... (d -> e)], a -> e
pipeline = functools.partial(functools.reduce, __ap)

# eof
