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

FILE_PATHS = {"HUMAN": ("data/ss.txt")}
args = {}

def get_ngram_vocab(n):
    words = [''.join(p) for p in itertools.product(ACIDS, repeat=n)]
    vocab = {}
    idx = 3
    for word in words:
        vocab[word] = idx
        idx += 1
    return vocab

def load_data(data_file, input_seq_to_vec, min_amino_acids=0, max_amino_acids=100000):
    X = []
    Y = []

    max_input_l = 0
    max_output_l = 0
    output_vocab = {'<START>': 1, '<END>': 2}
    output_vocab_idx = 3

    indices_to_exclude = []
    next_line_is_seq = False
    next_line_is_output = False

    with open(data_file, 'r') as f:
        for line_idx, line in enumerate(f):
            if 'A:sequence' in line:
                next_line_is_output = False
                next_line_is_seq = True
            elif 'A:secstr' in line:
                next_line_is_seq = False
                next_line_is_output = True
            elif (':sequence' in line) or ('secstr' in line):
                next_line_is_seq = False
                next_line_is_output = False
            elif next_line_is_seq: 
                x = input_seq_to_vec(line)
                if len(x) >= min_amino_acids and len(x) <= max_amino_acids:
                    max_input_l = max(len(x), max_input_l)
                else:
                    indices_to_exclude.append(line_idx)
                X.extend(x)
            elif next_line_is_output:
                y = []
                # start tag
                y.append(1)
                for output in line[:-1]:
                    if output not in output_vocab:
                        output_vocab[output] = output_vocab_idx
                        output_vocab_idx += 1
                    y.append(output_vocab[output])
                    max_output_l = max(max_output_l, len(y))
                # end tag
                y.append(2)
                Y.extend(y)

    X = remove_indices(X, indices_to_exclude)
    Y = remove_indices(Y, indices_to_exclude)

    return X, Y, output_vocab

def pad(vecs, padding_len, pad_char=0):
    for i in range(len(vecs)):
        vec = vecs[i]
        diff = max(0, padding_len - len(vec))
        padding = [pad_char for _ in range(diff)]
        vecs[i] = padding+vec
    return vecs

def remove_indices(data, indices_to_remove):
    return [d for i,d in enumerate(data) if i not in indices_to_remove]

# def load_output_data(output_data_file, cutoff=50):
#     output_vocab = {}
#     idx = 1
#     output_vecs = []
#     indices_to_remove = []
#     max_l = 0
#     with open(output_data_file, 'r') as f:
#         for line_idx, line in enumerate(f):
#             this_vec = []
#             for output in line.split():
#                 if output not in output_vocab:
#                     output_vocab[output] = idx
#                     idx += 1
#                 this_vec.append(output_vocab[output])
#             if len(this_vec) <= cutoff:
#                 max_l = max(max_l, len(this_vec))
#             else:
#                 indices_to_remove.append(line_idx)
#             output_vecs.append(this_vec)
#     return pad(output_vecs, max_l), indices_to_remove, len(output_vocab)


def ngram_seq_to_vec(sequence, n, vocab, start_tag=1, end_tag=2):
    vec = []

    vec.append(start_tag)

    for start_idx in range(len(sequence)-n):
        ngram = sequence[start_idx:start_idx+n]
        if ngram not in vocab:
            print ngram,
            print "not in vocab"
            assert False
        vec.append(vocab[ngram])

    vec.append(end_tag)

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
    data_file = FILE_PATHS[dataset]

    assert ngrams > 0
    vocab = get_ngram_vocab(ngrams)
    seq_to_vec = lambda s: ngram_seq_to_vec(s, ngrams, vocab)

    input_data, output_data, classes = load_data(data_file, seq_to_vec)
    train_input, train_output, test_input, test_output = split_data(input_data, output_data)

    print "Class | Index"
    for w, idx in classes.items():
        print w, idx

    print "Num inputs:",
    print len(input_data)

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
        f['nclasses'] = np.array([len(classes)], dtype=np.int32)
        f['start_idx'] = 1
        f['end_idx'] = 2

    print "Done."



if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
