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

import sys
import csv
import argparse
import traceback

import lib.cvutils as cvutils
import spot_larva

def munge(ns_str):
    return ns_str.\
        replace('Namespace', 'argparse.Namespace').\
        replace('Capture', 'cvutils.Capture')

def reify(ns_str):
    ns = eval(ns_str)
    try:
        ns.anchor_match_dist
    except AttributeError:
        # this argument was added
        ns.anchor_match_dist = spot_larva.default['anchor_match_dist']
        # the default value & units for this argument changed across versions
        ns.min_blob_area = ns.min_blob_area * ( 7.1 /  50.)
        ns.max_blob_area = ns.max_blob_area * (35.7 / 250.)
    return ns

def launch(ns_str):
    ns = reify(ns_str)
    with open(ns.movie.source + '_result.txt', 'w') as fd:
        print("Begin...", ns)

        # replace stdout
        stdout = sys.stdout
        sys.stdout = fd

        # run job
        try:
            spot_larva.main(ns)
        except KeyboardInterrupt:
            raise
        except:
            tb = traceback.format_exc()
            sys.stdout.write(tb)
            sys.stderr.write(tb)
        # restore stdout
        sys.stdout = stdout
        print("End...", ns)

def main(args):
    ns_strs = [munge(s) for _, s in csv.reader(args.args_file)]
    for s in ns_strs:
        print(reify(s))
    assert raw_input("Okay? [y/N] ") in {'y', 'yes'}
    print('Replaying...')
    for s in ns_strs:
        launch(s)

if __name__ == '__main__':
    # args
    p = argparse.ArgumentParser()
    p.add_argument \
        ( 'args_file'
        , type = open
        , help = '''Containing args Namespace representations for spot_larva's main''')
    # main
    exit(main(p.parse_args()))

# eof
