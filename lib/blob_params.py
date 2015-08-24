from __future__ import \
    ( nested_scopes
    , generators
    , division
    , absolute_import
    , with_statement
    , print_function
    , unicode_literals
    )

import random

template = lambda: \
    { 'minRepeatability':(long(0), long(10))
    , 'minThreshold':(0.0, 'maxThreshold')
    , 'maxThreshold':('minThreshold', 255.0)
    , 'thresholdStep':(5.0, 25.0)
    , 'minDistBetweenBlobs':(1.0, 400.0)
    , 'filterByArea':True
    , 'minArea':(1.0, 'maxArea')
    , 'maxArea':('minArea', 400.0)
    , 'filterByCircularity':True
    , 'minCircularity':(0.0, 'maxCircularity')
    , 'maxCircularity':('minCircularity', 1.0)
    , 'filterByColor':True
    , 'blobColor':(0, 255)
    , 'filterByConvexity':True
    , 'minConvexity':(0.0, 'maxConvexity')
    , 'maxConvexity':('minConvexity', 1.0)
    , 'filterByInertia':True
    , 'minInertiaRatio':(0.0, 'maxInertiaRatio')
    , 'maxInertiaRatio':('minInertiaRatio', 1.0)
    }.copy()

def shuffled(xs_):
    xs = xs_[:]
    random.shuffle(xs)
    return xs

def randIn(lb, ub):
    '''random from [lb, ub) interval'''
    if isinstance(lb, long):
        return long(random.randrange(lb, ub))
    if isinstance(lb, int):
        return random.randrange(lb, ub)
    elif isinstance(lb, float):
        return random.random() * (ub - lb) + lb
    else:
        raise

indexOrWhole = lambda v, i: \
        v[i] if isinstance(v, (list, tuple)) else v

def paramRange(pd, k):
    lb, ub = pd[k]
    if isinstance(lb, basestring):
        return indexOrWhole(pd[lb], 0), ub
    elif isinstance(ub, basestring):
        return lb, indexOrWhole(pd[ub], 1)
    else:
        return pd[k]

def randParam(pd, k):
    if isinstance(pd[k], tuple):
        lb, ub = paramRange(pd, k)
        pd[k] = randIn(lb, ub)
    return pd

def filterStatus(pd, fks, k):
    fs = filter(lambda fk: fk in k, fks)
    assert len(fs) in {0, 1}, 'exactly one filter may apply to a key, got {} for {}'.format(fs, repr(k))
    if fs:
        fk, = fs
        return pd['filterBy' + fk]
filterStatus.fks = lambda pd: [k.replace('filterBy', '') for k, v in pd.items() if isinstance(v, bool)]

####

def randParams(params_or_ranges=None):
    pd = template()
    params_or_ranges and pd.update(params_or_ranges)
    #
    return reduce(randParam, shuffled(pd.keys()), pd)

def paramRanges(params_or_ranges=None):
    '''import lib.blob_params as b; print(b.paramRanges())'''
    pd = template()
    params_or_ranges and pd.update(params_or_ranges)
    #
    return {k: (paramRange(pd, k) if isinstance(v, tuple) else v)
            for k, v in pd.items()}

def paramBuckets(bucketCount, params_or_ranges=None):
    '''import lib.blob_params as b; bf = b.paramBuckets(2, b.paramRanges()); print({k: ((bf[k](m_m[0]), bf[k](m_m[1] - 0.0001)) if isinstance(m_m, tuple) else m_m) for k, m_m in b.paramRanges().items()})'''
    pd = template()
    params_or_ranges and pd.update(params_or_ranges)
    #
    def bucketFn(lb, ub):
        width = (ub - lb) / bucketCount
        def f(x):
            assert lb <= x < ub, 'lb={} <= x={} < ub={}'.format(lb, x, ub)
            return int((x - lb) // width)
        return f
    const = lambda _: -1
    return {k: (bucketFn(v[0], v[1]) if isinstance(v, tuple) else const)
            for k, v in pd.items()}

def trainingState(bucketCount, params_or_ranges=None):
    '''import lib.blob_params as b; print(b.trainingState(0, {"filterByInertia": False, "filterByConvexity": False, "filterByColor":False, "filterByCircularity":False, "filterByArea":False}))'''
    # produce training state arrays

    # should have an index of property names which is in-order of the axes in the arr
    # index should include properties which are always-on or have a 'filterBy...':True

    # arr should have an axis for each param
    # each axis length is equal to bucketCount
    # there is one extra axis for data (length 2)

    # arr[3,4,2,6,:] == [count, avg_error]
    # ^ 3 parameters
    #   bucket count is at least 7
    #   this model's params fell into buckets 3,4,2 and 6
    #   count is the number of models that fell in these buckets
    #   avg_error is the average error up until now
    pd = template()
    params_or_ranges and pd.update(params_or_ranges)
    #
    fks = filterStatus.fks(pd)
    keyStatus = {k: filterStatus(pd, fks, k) for k in pd}
    index = filter \
        ( lambda k: ( 'filterBy' not in k \
                  and keyStatus[k] in {True, None} \
                  and isinstance(pd[k], tuple)
                    )
        , sorted \
            ( pd.keys()
            , key = lambda k: ('_' + k) if keyStatus[k] is None else k ) )
    print('Training {} parameters across {} buckets: at least {} states'.format(
        len(index), bucketCount, bucketCount ** len(index)))
    return index, len(index) * [bucketCount]

# eof
