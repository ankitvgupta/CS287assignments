#!/usr/bin/env python

"""Part-Of-Speech Preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re
import codecs
import string

# Your preprocessing, features construction, and word2vec code.

legal_chars = list(string.printable)+['<space>', '</s>', '</s>\n']
VOCAB = dict([(v, k+1) for k, v in enumerate(legal_chars)])

FILE_PATHS = {"PTB": ("data/train_chars.txt",
                      "data/valid_chars.txt",
                      "data/test_chars.txt"
                      )}
args = {}

def file_to_input(file_path):
    space_index = VOCAB['<space>']
    with open(file_path, 'r') as f:
        full_sample = f.next()
        all_chars = full_sample.split(' ')
        X = np.array([VOCAB[c] for c in all_chars])
        spaces = np.argwhere(X==space_index)
        next_spaces = spaces-1
        Y = np.zeros(X.shape, dtype=int)
        Y[next_spaces] = 1
        return X, Y



def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set",
                        type=str, default='PTB', nargs='?')
    args = parser.parse_args(arguments)
    dataset = args.dataset
    train, valid, test = FILE_PATHS[dataset]

    train_input, train_output = file_to_input(train)
    valid_input, valid_output = file_to_input(valid)

    # TODO
    test_input = np.array([])

    V = len(VOCAB)
    C = 2

    filename = args.dataset + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_input'] = train_input
        f['train_output'] = train_output
        if valid:
            f['valid_input'] = valid_input
            f['valid_output'] = valid_output
        if test:
            f['test_input'] = test_input
        f['nfeatures'] = np.array([V], dtype=np.int32)
        f['nclasses'] = np.array([C], dtype=np.int32)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
