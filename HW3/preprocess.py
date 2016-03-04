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
            if line[0] == 'Q':
                options = parse_options(line[2:], vocab)
                Os.append(options)
            elif line[0] == 'C':
                context = parse_context(line[2:], vocab, dwin)
                Cs.append(context)
            else:
                assert False

    return Cs, Os

def create_output_data(valid, valid_blanks, vocab_dict):
    Y = []
    with open(valid, 'rb') as vf:
        with open(valid_blanks, 'rb') as vb:
            for blanks_line in vb:
                blank_split = blanks_line.split()
                if blank_split[0] == 'C':
                    context = blank_split[1:]
                    full_text = vf.next()
                    full_split = full_text.split()
                    full_split += ['</s>']

                    # if the context is the full sentence, predict another </s>
                    if context[-1] == '</s>':
                        full_split.append('</s>')

                    context_len = len(context)
                    # debugging
                    assert context_len < len(full_split)
                    for i, b in enumerate(context):
                        assert (full_split[i] == b) or (full_split[i] == '<unk>'), full_split

                    target = full_split[context_len]
                    y = vocab_dict[target]
                    Y.append(y)

    return Y


FILE_PATHS = {"PTB": ("data/train.txt",
                      "data/valid.txt",
                      "data/valid_blanks.txt",
                      "data/test_blanks.txt",
                      "data/words.dict"),
              "SMALL": ("data/train.1000.txt",
                       "data/valid.1000.txt",
                       "data/valid_blanks.txt",
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
    parser.add_argument('allwins', help="If true, generate all window sizes <= dwin",
                        type=bool, default=True, action='store', nargs='?')

    args = parser.parse_args(arguments)
    dataset = args.dataset
    dwin = args.dwin
    allwins = args.allwins
    train, valid, valid_blanks, test_blanks, word_dict = FILE_PATHS[dataset]

    # Load vocab dict
    print "Loading vocab dict..."
    vocab_dict = load_vocab_dict(word_dict)

    numFeatures = len(vocab_dict)
    numClasses = len(vocab_dict)

    print "Creating validation output..."
    valid_true_outs = create_output_data(valid, valid_blanks, vocab_dict)

    if allwins:
        dwins = range(0, dwin+1)
    else:
        dwins = [dwin]

    for dw in dwins:

        print "Creating language samples from training data..."
        train_input, train_output = create_training_data(train, vocab_dict, dw)

        print "Creating language samples from valid data..."
        valid_context, valid_options = create_data_from_blanks(valid_blanks, vocab_dict, dw)

        print "Creating language samples from test data..."
        test_context, test_options = create_data_from_blanks(test_blanks, vocab_dict, dw)

        # Write out to hdf5
        filename = args.dataset + '_'+str(dw)+'.hdf5'
        print "Writing out to "+filename
        with h5py.File(filename, "w") as f:
            f['train_input'] = train_input
            f['train_output'] = train_output
            if valid_blanks:
                f['valid_context'] = valid_context
                f['valid_options'] = valid_options
                f['valid_true_outs'] = valid_true_outs
            if test_blanks:
                f['test_context'] = test_context
                f['test_options'] = test_options

            f['numFeatures'] = np.array([numFeatures], dtype=np.int32)
            f['numClasses'] = np.array([numClasses], dtype=np.int32)
            f['d_win'] = np.array([dw], dtype=np.int32)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
