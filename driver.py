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
        print("Begin:")
        print(ns)
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

def main(args):
    print(args)
    # obtain ns_strs
    if args.args_file:
        print('Reading', args.args_file.name)
        ns_strs = [munge(s) for _, s in csv.reader(args.args_file)]
    else:
        print('Reading stdin')
        ns_strs = [munge(ln.strip()) for ln in sys.stdin]
    # visually verify
    print("Reifying...")
    for s in ns_strs:
        print(reify(s))
    # ask for confirmation
    if args.interactive and args.args_file:
        assert raw_input("Okay? [y/N] ") in {'y', 'yes'}
    # replay
    print('Replaying...')
    for s in ns_strs:
        launch(s)

if __name__ == '__main__':
    # args
    p = argparse.ArgumentParser(description='''Run (or re-run) multiple
    spot_larva programs, storing the output in *_result.txt. If no --args-file
    option is given, then read Namespace representations from stdin.''')
    p.add_argument \
        ( '--args-file'
        , type = open
        , metavar = 'path'
        , help = '''a `meta_args.txt` file as output by grepping `*_result.csv`
        files for args which contains Namespace representations for
        spot_larva's main function.''')
    p.add_argument \
        ( '--non-interactive'
        , action = 'store_false'
        , dest = 'interactive'
        , help = '''Do not ask for confirmation after reifying the Namespaces
        before starting the replay. Implied when no --args-file option is
        given.''')
    # main
    exit(main(p.parse_args()))

# eof
