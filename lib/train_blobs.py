#!/usr/bin/env python2
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
import pickle

import cv2
import numpy

import lib.blob_params as blob_params

TrainSession = collections.namedtuple('TrainSession',
        'bucket_count param_space param_index train_state')

def restore_session(file_path):
    with open('{}.p'.format(file_path), 'rb') as p_file:
        bucket_count, param_space, param_index, train_state_path = pickle.load(p_file)
    with open(train_state_path, 'rb') as npy_file:
        train_state = numpy.load(npy_file)
    assert isinstance(bucket_count, int)
    assert isinstance(param_space, dict)
    assert isinstance(param_index, list)
    assert isinstance(train_state, numpy.ndarray)
    return TrainSession \
            ( bucket_count = bucket_count
            , param_space = param_space
            , param_index = param_index
            , train_state = train_state
            )

def preserve_session(file_path, sess, paranoid=False):
    assert isinstance(sess.bucket_count, int)
    assert isinstance(sess.param_space, dict)
    assert isinstance(sess.param_index, list)
    assert isinstance(sess.train_state, numpy.ndarray)
    with open('{}.npy'.format(file_path), 'wb') as npy_file:
        numpy.save(npy_file, sess.train_state)
    with open('{}.p'.format(file_path), 'wb') as p_file:
        pickle.dump \
                ( ( sess.bucket_count
                  , sess.param_space
                  , sess.param_index
                  , npy_file.name
                  )
                , p_file
                )
    if paranoid:
        sessB = restore_session(file_path)
        assert sess.bucket_count == sessB.bucket_count
        assert sess.param_space == sessB.param_space
        assert sess.param_index == sessB.param_index
        assert (sess.train_state == sessB.train_state).all()

def new_session(bucket_count, param_space):
    param_index = blob_params.new_index(param_space)
    train_state = numpy.zeros(len(param_index) * [bucket_count] + [3])
    return TrainSession \
            ( bucket_count = bucket_count
            , param_space = param_space
            , param_index = param_index
            , train_state = train_state
            )

def print_session_info(sess):
    print('Training {} parameters across {} buckets. There are at least {} states.'.format \
            ( len(sess.param_index)
            , sess.bucket_count
            , sess.bucket_count ** len(sess.param_index)
            ))
    #print('items', sess.train_state.size / 3)
    #print('items', sess.train_state[...,0].size)
    print('Training state uses {} MiB ({:,} bytes).'.format \
            ( sess.train_state.size * sess.train_state.itemsize / 1024 / 1024
            , sess.train_state.size * sess.train_state.itemsize
            ))

def training_status(sess, msg, verbose=False):
    counts = sess.train_state[..., 0]
    explored = (counts > 0).sum() / counts.size
    explored_10times = (counts >= 10).sum() / counts.size
    print( '{:.3}/1 paths explored;'.format(explored)
         , '{:.3}/1 paths explored 10 times'.format(explored_10times)
         , msg)
    # do we have enough information to calculate gradient for each feature and adjust the rand window?
    return explored > 0.95

def train \
        ( sess
        , examples # iter<(ndarray, expect)>
        # FIXME: put error into TrainSession
        , error # expect, [KeyPoint] -> num
        , verbose = False
        , imshow_window = None
        ):
    rangesd = blob_params.paramRanges(sess.param_space)
    bucketd = blob_params.paramBuckets(sess.bucket_count, rangesd)
    counts = sess.train_state[..., 0]
    signed_err_sums = sess.train_state[..., 1] # good for ?
    unsigned_err_sums = sess.train_state[..., 2] # good for gradient descent

    for i, (im, label) in enumerate(examples):
        if verbose:
            print('frame', i)

        # choose random params
        paramd = blob_params.randParams(sess.param_space)
        if verbose:
            print('\n'.join('  {}: {}'.format(k, paramd[k]) for k in sess.param_index))

        # make a detector and detect blobs in the frame
        blobs = blob_params.mkDetector(paramd, verbose=verbose and 0 == i).detect(im)

        # calculate error from expected and actual
        signed_err = error(label, blobs)
        if verbose:
            print('-> signed_err', signed_err)

        # bucketize chosen param values
        path = tuple(bucketd[k](paramd[k]) for k in sess.param_index)
        if verbose:
            print('-> bucket path', path)

        # update state
        counts[path] += 1
        signed_err_sums[path] += signed_err
        unsigned_err_sums[path] += abs(signed_err)

        if imshow_window:
            cv2.drawKeypoints(im, blobs, im, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow(imshow_window, im)
            cv2.waitKey(1)

# eof
