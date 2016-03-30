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

legal_chars = ['<space>', '</s>', '</s>\n']+list(string.printable)
VOCAB = dict([(v, k+1) for k, v in enumerate(legal_chars)])

FILE_PATHS = {"PTB": ("data/train_chars.txt",
                      "data/valid_chars.txt",
                      "data/test_chars.txt"
                      )}
args = {}

def file_to_input(file_path, remove_spaces=False):
    space_index = VOCAB['<space>']
    with open(file_path, 'r') as f:
        full_sample = f.next()
        all_chars = full_sample.split(' ')
        X = np.array([VOCAB[c] for c in all_chars])
        spaces = np.argwhere(X==space_index)
        next_spaces = spaces-1
        Y = np.ones(X.shape, dtype=int)
        Y[next_spaces] = 2
        if remove_spaces:
            X = np.delete(X, spaces)
            Y = np.delete(Y, spaces)
        return X, Y

def test_file_to_input(file_path, padding_char='</s>'):
    space_index = VOCAB['<space>']
    max_chars = 0
    char_sequences = []
    X = []
    with open(file_path, 'r') as f:
        for sequence in f:
            all_chars = sequence.split(' ')
            max_chars = max(len(all_chars), max_chars)
            char_sequences.append(all_chars)

        for char_sequence in char_sequences:
            # pad sequence
            diff = max_chars - len(char_sequence)
            for _ in range(diff):
                char_sequence.append(padding_char)

            x = np.array([VOCAB[c] for c in char_sequence])            
            X.append(x)
        
        return np.array(X)



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
    valid_input, valid_output = file_to_input(valid, remove_spaces=True)

    test_input = test_file_to_input(test)

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
        f['spaceIdx'] = np.array([VOCAB['<space>']], dtype=np.int32)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
