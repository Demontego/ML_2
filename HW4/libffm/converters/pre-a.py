#!/usr/bin/env python3

import argparse, csv, sys

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('csv_path', type=str)
parser.add_argument('dense_path', type=str)
parser.add_argument('sparse_path', type=str)
args = vars(parser.parse_args())

#These features are dense enough (they appear in the dataset more than 4 million times), so we include them in GBDT
target_cat_feats =  ['C15-', 'C6-995', 'C11-0', 'C9-15', 'C14-', 'C7-2', 'C7-0', 'C12-106', 'C12-105', 'C12-104', 'C5-30', 'C11-1', 'C10-821', 'C10-802', 'C5-88', 'C8-177', 'C5-10', 'C3-2252', 'C8-452', 'C12-103', 'C8-293', 'C8-419', 'C8-428', 'C10-49', 'C5-11', 'C5-54', 'C8-8', 'C8-341', 'C8-361']

with open(args['dense_path'], 'w') as f_d, open(args['sparse_path'], 'w') as f_s:
    for row in csv.DictReader(open(args['csv_path'])):
        feats = []
        for j in range(1, 3):
            val = row['I{0}'.format(j)]
            if val == '':
                val = -10 
            feats.append('{0}'.format(val))
        f_d.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
        
        cat_feats = set()
        for j in range(1, 16):
            field = 'C{0}'.format(j)
            key = field + '-' + row[field]
            cat_feats.add(key)

        feats = []
        for j, feat in enumerate(target_cat_feats, start=1):
            if feat in cat_feats:
                feats.append(str(j))
        f_s.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
