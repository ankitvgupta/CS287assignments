#!/usr/bin/env python

"""Language modeling preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re
import codecs

# Your preprocessing, features construction, and word2vec code.

def load_vocab_dict(file_path):
    vocab_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            idx, word, _ = line.strip().split()
            vocab_dict[word] = int(idx)
    m = max(vocab_dict.values())
    vocab_dict['<s>'] = m+1
    vocab_dict['</s>'] = m+2

    return vocab_dict

def parse_line(s, vocab, dwin, frontpad=True, backpad=True, lowercase=False):
    s = s.strip()
    if lowercase:
        s = s.lower()

    if frontpad:
        for _ in range(dwin):
            s = '<s> ' + s

    if backpad:
        s = s + ' </s>'

    idxs = []

    for word in s.split():
        if word not in vocab:
            word = '<unk>'
        idxs.append(vocab[word])

    return idxs

def parse_options(s, vocab):
    return parse_line(s, vocab, 0, frontpad=False, backpad=False)

def parse_context(s, vocab, dwin):
    full_context = parse_line(s, vocab, dwin, backpad=False)
    start = len(full_context)-dwin
    return full_context[start:]

def split_idxs(idxs, dwin):
    X = []
    Y = []

    l = len(idxs)
    target = dwin

    while target < l:
        x = idxs[target-dwin:target]
        y = idxs[target]

        X.append(x)
        Y.append(y)

        target += 1

    return X, Y

def create_training_data(data_file, vocab, dwin):
    Xs = []
    Ys = []

    with open(data_file, 'rb') as f:
        for line in f:
            sentence = parse_line(line, vocab, dwin)
            X, Y = split_idxs(sentence, dwin)
            Xs.extend(X)
            Ys.extend(Y)

    return Xs, Ys

def create_data_from_blanks(data_file, vocab, dwin):
    Cs = []
    Os = []

    with open(data_file, 'rb') as f:
        for line in f:
            line_split = line.split()
            if line[0] == 'Q':
                options = parse_options(line[1:], vocab)
                Os.append(options)
            elif line[0] == 'C':
                context = parse_context(line[1:], vocab, dwin)
                Cs.append(context)
            else:
                assert False

    return Cs, Os

# TODO
def create_output_data(data_file):
    pass

FILE_PATHS = {"PTB": ("data/train.txt",
                      "data/valid.txt",
                      "data/valid_blanks.txt",
                      "data/valid_outs.txt",
                      "data/test_blanks.txt",
                      "data/words.dict"),
              "SMALL": ("data/train.1000.txt",
                       "data/valid.1000.txt",
                       "data/valid_blanks.txt",
                       "data/valid_outs.txt",
                       "data/test_blanks.txt",
                       "data/words.1000.dict")}
args = {}


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set",
                        type=str)
    parser.add_argument('dwin', help="Window size",
                        type=int)
    args = parser.parse_args(arguments)
    dataset = args.dataset
    dwin = args.dwin
    train, valid, valid_blanks, valid_outs, test_blanks, word_dict = FILE_PATHS[dataset]

    # Force window size odd
    assert dwin > 0 and dwin % 2 == 1

    # Load vocab dict
    print "Loading vocab dict..."
    vocab_dict = load_vocab_dict(word_dict)

    numFeatures = len(vocab_dict)
    numClasses = len(vocab_dict)

    print "Creating language samples from training data..."
    train_input, train_output = create_training_data(train, vocab_dict, dwin)

    print "Creating language samples from valid data..."
    valid_context, valid_options = create_data_from_blanks(valid_blanks, vocab_dict, dwin)

    print "Creating language samples from test data..."
    test_context, test_options = create_data_from_blanks(test_blanks, vocab_dict, dwin)

    # TODO ONCE FILE IS ADDED
    # print "Creating validation output..."
    # valid_distribution = create_output_data(valid_output)

    # Write out to hdf5
    print "Writing out to hdf5"
    filename = args.dataset + '_'+str(dwin)+'.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_input'] = train_input
        f['train_output'] = train_output
        if valid_blanks:
            f['valid_context'] = valid_context
            f['valid_options'] = valid_options
            # f['valid_distribution'] = valid_distribution
        if test_blanks:
            f['test_context'] = test_context
            f['test_options'] = test_options

        f['numFeatures'] = np.array([numFeatures], dtype=np.int32)
        f['numClasses'] = np.array([numClasses], dtype=np.int32)
        f['d_win'] = np.array([dwin], dtype=np.int32)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
