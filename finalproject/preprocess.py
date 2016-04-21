#!/usr/bin/env python

"""Final Project Preprocessing
"""
from matplotlib import pyplot as plt

import argparse
import string
import sys
import itertools
import numpy as np
import h5py

ACIDS = list(string.ascii_uppercase)

FILE_PATHS = {"HUMAN": ("data/human.pr.filter",
                       "data/human.go.filter"),
              "ALL": (None,
                       None)}
args = {}

def get_ngram_vocab(n):
    words = [''.join(p) for p in itertools.product(ACIDS, repeat=n)]
    vocab = {}
    idx = 1
    for word in words:
        vocab[word] = idx
        idx += 1
    return vocab

def load_input_data(input_data_file, seq_to_vec, cutoff=2000):
    X = []
    max_l = 0
    indices_to_remove = []
    with open(input_data_file, 'r') as f:
        for line_idx, line in enumerate(f):
            x = seq_to_vec(line)
            if len(x) <= cutoff:
                max_l = max(len(x), max_l)
            else:
                indices_to_remove.append(line_idx)
            X.append(x)
    return pad(X, max_l), indices_to_remove

def pad(vecs, padding_len, pad_char=0):
    for i in range(len(vecs)):
        vec = vecs[i]
        diff = max(0, padding_len - len(vec))
        padding = [pad_char for _ in range(diff)]
        vecs[i] = padding+vec
    return vecs

def remove_indices(data, indices_to_remove):
    return [d for i,d in enumerate(data) if i not in indices_to_remove]

def load_output_data(output_data_file, cutoff=50):
    output_vocab = {}
    idx = 1
    output_vecs = []
    indices_to_remove = []
    max_l = 0
    with open(output_data_file, 'r') as f:
        for line_idx, line in enumerate(f):
            this_vec = []
            for output in line.split():
                if output not in output_vocab:
                    output_vocab[output] = idx
                    idx += 1
                this_vec.append(output_vocab[output])
            if len(this_vec) <= cutoff:
                max_l = max(max_l, len(this_vec))
            else:
                indices_to_remove.append(line_idx)
            output_vecs.append(this_vec)
    return pad(output_vecs, max_l), indices_to_remove, len(output_vocab)


def ngram_seq_to_vec(sequence, n, vocab):
    vec = []

    for start_idx in range(len(sequence)-n):
        ngram = sequence[start_idx:start_idx+n]
        if ngram not in vocab:
            print ngram,
            print "not in vocab"
            assert False
        vec.append(vocab[ngram])

    return vec


def split_data(input_data, output_data, split_perc=0.8):
    l = len(input_data)
    assert len(output_data) == l
    index_shuf = range(l)
    input_shuf = [input_data[i] for i in index_shuf]
    output_shuf = [output_data[i] for i in index_shuf]
    cutoff = int(split_perc*l)
    return input_shuf[:cutoff], output_shuf[:cutoff], input_shuf[cutoff:], output_shuf[cutoff:]
    

def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set", default="HUMAN",
                        type=str, nargs='?')
    parser.add_argument('--ngrams', help="N gram length", default=1,
                        type=int, nargs='?')
    args = parser.parse_args(arguments)
    dataset = args.dataset
    ngrams = args.ngrams
    input_data_file, output_data_file = FILE_PATHS[dataset]

    assert ngrams > 0
    vocab = get_ngram_vocab(ngrams)
    seq_to_vec = lambda s: ngram_seq_to_vec(s, ngrams, vocab)

    input_data, input_indices_to_remove = load_input_data(input_data_file, seq_to_vec)
    output_data, output_indices_to_remove, nclasses = load_output_data(output_data_file)

    indices_to_remove = set(input_indices_to_remove) | set(output_indices_to_remove)
    input_data = remove_indices(input_data, indices_to_remove)
    output_data = remove_indices(output_data, indices_to_remove)

    train_input, train_output, test_input, test_output = split_data(input_data, output_data)

    # Write out to hdf5
    print "Writing out to hdf5"
    filename = args.dataset + '.hdf5'
    with h5py.File(filename, "w") as f:

        f['train_input'] = np.array(train_input, dtype=np.int32)
        f['train_output'] = np.array(train_output, dtype=np.int32)
        f['test_input'] = np.array(test_input, dtype=np.int32)
        f['test_output'] = np.array(test_output, dtype=np.int32)
        f['ngrams'] = np.array([ngrams], dtype=np.int32)
        f['vocab_size'] = np.array([len(vocab)], dtype=np.int32)
        f['nclasses'] = np.array([nclasses], dtype=np.int32)

    print "Done."



if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
