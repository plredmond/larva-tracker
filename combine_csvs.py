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

import os.path
import argparse
import csv


def load_one(csv_fd):
    section_keywords = ['distance traveled', 'cumulative distance', 'average speed', 'time bounds']
    output = {'metadata':{}}
    section = None
    cols = None
    rows = None
    for cells in filter(None, csv.reader(csv_fd, dialect='excel')):
        if section_keywords and section_keywords[0] in cells[0].lower():
            if rows is None and section is not None:
                rows = len(output[section][1])
            section = section_keywords.pop(0)
            output[section] = (cells, [])
        elif section is not None:
            if cols is None:
                cols = len(cells)
            assert cols == len(cells), 'all section rows have same number of columns'
            output[section][1].append(cells)
        else:
            header, data = cells
            assert header not in output['metadata'], 'all metadata headers are unique'
            output['metadata'][header] = data
    assert all(len(output[k][1]) == rows for k in output.viewkeys() - set(['metadata'])), 'all sections have same number of rows'
    assert not section_keywords, 'all sections must be found'
    return output


def main(args):
    path, _ = os.path.split(args.filepaths[0])
    assert all(path == os.path.split(fp)[0] for fp in args.filepaths), 'all csvs live in the same directory'
    csvs = [load_one(open(fp, 'rb')) for fp in sorted(args.filepaths)]
    assert all(csvs[0].viewkeys() == c.viewkeys() for c in csvs), 'all csvs have same keys'
    assert all(csvs[0]['metadata'].viewkeys() == c['metadata'].viewkeys() for c in csvs), 'all csvs have same meta keys'
    print('outputting to', path)
    assert raw_input('y/n> ') in {'y', 'yes', 'Y', 'Yes'}
    with open(os.path.join(path, 'overall-meta.csv'), mode='wb') as fd:
        w = csv.DictWriter(fd, sorted(csvs[0]['metadata'].keys(), key=lambda k: len(csvs[0]['metadata'][k])), dialect='excel')
        w.writeheader()
        w.writerows(c['metadata'] for c in csvs)
    print('Wrote', fd.name)
    for section in sorted(csvs[0].viewkeys() - set(['metadata'])):
        with open(os.path.join(path, 'overall-{}.csv'.format('_'.join(section.split()))), mode='wb') as fd:
            w = csv.writer(fd, dialect='excel')
            for c in csvs:
                w.writerow([c['metadata']['file']])
                w.writerow(c[section][0])
                w.writerows(c[section][1])
                print(c['metadata']['file'], end=' ')
        print()
        print('Wrote', fd.name)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument \
        ( 'filepaths'
        , metavar = 'csv'
        , nargs = '+'
        )
    exit(main(p.parse_args()))

# eof

