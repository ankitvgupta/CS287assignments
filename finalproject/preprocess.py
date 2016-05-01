#!/usr/bin/env python

"""Final Project Preprocessing
"""
from matplotlib import pyplot as plt
from princfilter import pfilter

import argparse
import string
import sys
import itertools
import numpy as np
import h5py

ACIDS = {'<': 1, '>': 2, 'A': 3, 'C': 4, 'E': 5, 'D': 6, 'G': 7, 'F': 8, 'I': 9, 'H': 10, 'K': 11, 'M': 12, 'L': 13, 'N':14, 'Q':15, 'P':16, 'S':17, 'R':18, 'T':19, 'W':20, 'V':21, 'Y':22, 'X':23, 'U': 24, 'Z': 25, 'B':26, 'O':27}
LABELS = {'<': 1, '>': 2, 'L': 3, ' ': 3, 'B': 4, 'E': 5, 'G': 6, 'I': 7, 'H': 8, 'S': 9, 'T': 10}

FILE_PATHS = {"HUMAN": ("data/ss.txt"), 
              "FILT": ("data/ss_filtered_input.txt", "data/ss_filtered_output.txt"),
              "CB513": ("data/cb513+profile_split1.npy"), 
              "PRINC": ("data/cullpdb+profile_6133_filtered.npy")}
args = {}

# return two lists of amino acid / label strings
def parse_human(data_file):
    X_strings = ''
    Y_strings = ''

    next_line_is_seq = False
    next_line_is_output = False

    with open(data_file, 'r') as f:
        for line in f:
            if 'A:sequence' in line:
                next_line_is_output = False
                next_line_is_seq = True
                X_strings = X_strings + '<'
            elif 'A:secstr' in line:
                next_line_is_seq = False
                next_line_is_output = True
                Y_strings = Y_strings + '<'
            elif (':sequence' in line) or ('secstr' in line):
                next_line_is_seq = False
                next_line_is_output = False
            elif next_line_is_seq: 
                X_strings = X_strings + line[:-1]
            elif next_line_is_output:
                Y_strings = Y_strings + line[:-1]

    splitX = X_strings.split('<')[1:]
    splitY = Y_strings.split('<')[1:]

    assert len(splitX) == len(splitY)

    return splitX, splitY

def parse_filtered_human(files):
    input_file, output_file = files

    X_strings = []
    Y_strings = []

    with open(input_file, 'r') as f1:
        for line in f1:
            X_strings.append(line[:-1])

    with open(output_file, 'r') as f2:
        for line in f2:
            Y_strings.append(line[:-1])


    print len(X_strings), len(Y_strings)
    assert len(X_strings) == len(Y_strings)

    return X_strings, Y_strings

# return two lists of amino acid / label strings
def parse_princeton(data_file, num_proteins):
    X_strings = []
    Y_strings = []

    # NoSeq is padding
    acid_order = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','NoSeq']
    label_order = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T','NoSeq']

    data = np.load(data_file)
    amino_acids = np.reshape(data, (num_proteins, 700, 57))
    
    for p in range(num_proteins):
        x = ''
        y = ''
        for a in range(700):
            # reached end of this sequence
            if amino_acids[p][a][21]:
                # label should also be noseq
                assert amino_acids[p][a][30]
                X_strings.append(x)
                Y_strings.append(y)
                break
            else:
                this_acids = np.nonzero(amino_acids[p][a][:22])[0]
                assert len(this_acids) == 1
                this_acid = acid_order[this_acids[0]]
                this_labels = np.nonzero(amino_acids[p][a][22:30])[0]
                assert len(this_labels) == 1
                this_label = label_order[this_labels[0]]

                x = x+this_acid
                y = y+this_label


    return X_strings, Y_strings

def ngram_encoder(vocab_dict, ngram, start_pad='<', end_pad='>'):
    halfwin = (ngram-1)/2
    def this_encoder(s):
        padded_str = ''.join([start_pad for _ in range(halfwin)])+s+''.join([end_pad for _ in range(halfwin)])
        encoding = []
        for idx in range(halfwin, len(s)+halfwin):
            start_idx = idx-halfwin
            end_idx = idx+halfwin+1
            this_encoding = []
            for i in range(start_idx, end_idx):
                c = padded_str[i]
                this_encoding.append(vocab_dict[c])
            encoding.append(this_encoding)
        return encoding
    return this_encoder


def encode_strings(str_list, string_encoder):
    encoded = []

    for s in str_list:
        encoded.extend(string_encoder(s))

    return encoded


def split_data(input_data, output_data, split_perc=0.8):
    l = len(input_data)
    assert len(output_data) == l
    cutoff = int(split_perc*l)
    return input_data[:cutoff], output_data[:cutoff], input_data[cutoff:], output_data[cutoff:]
 

def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('train_dataset', help="Train data set", default="HUMAN",
                        type=str, nargs='?')
    # if None, then train dataset is split into train and test
    parser.add_argument('--test', help="Test data set", default=None,
                        type=str, nargs='?')
    parser.add_argument('--dwin', help="Train data set", default=1,
                        type=int, nargs='?')
    args = parser.parse_args(arguments)
    train_dataset = args.train_dataset
    test_dataset = args.test
    dwin = args.dwin

    assert dwin % 2 == 1

    if test_dataset is not None and test_dataset != "CB513": 
        print "Test dataset should only be CB513 as of now."
        assert False

    train_data_file = FILE_PATHS[train_dataset]
    if train_dataset == "CB513":
        X_strings, Y_strings = parse_princeton(train_data_file, 514)
    elif train_dataset == "PRINC":
        X_strings, Y_strings = parse_princeton(train_data_file, 5534)
    elif train_dataset == "HUMAN":
        X_strings, Y_strings = parse_human(train_data_file)
    elif train_dataset == "FILT":
        X_strings, Y_strings = parse_filtered_human(train_data_file)
    else:
        print "Unknown train dataset", train_dataset
        assert False

    input_encoder = ngram_encoder(ACIDS, dwin)
    output_encoder = ngram_encoder(LABELS, 1)

    input_data = encode_strings(X_strings, input_encoder)
    output_data = encode_strings(Y_strings, output_encoder)

    if test_dataset is None:
        train_input, train_output, test_input, test_output = split_data(input_data, output_data)
    else:
        train_input = input_data
        train_output = output_data

        test_data_file = FILE_PATHS[test_dataset]
        if test_dataset == "CB513":
            X_strings, Y_strings = parse_princeton(test_data_file, 514)
        else:
            print "This should only be CB513..."
            assert False

        test_input = encode_strings(X_strings, input_encoder)
        test_output = encode_strings(Y_strings, output_encoder)


    # Write out to hdf5
    filename = train_dataset
    if test_dataset is not None: filename = filename + '_'+test_dataset
    filename = filename+'_'+str(dwin)
    filename = filename+'.hdf5'
    print "Writing out to", filename
    with h5py.File(filename, "w") as f:

        f['train_input'] = np.array(train_input, dtype=np.int32)
        f['train_output'] = np.array(np.squeeze(train_output), dtype=np.int32)
        f['test_input'] = np.array(test_input, dtype=np.int32)
        f['test_output'] = np.array(np.squeeze(test_output), dtype=np.int32)
        f['vocab_size'] = np.array([max(ACIDS.values())], dtype=np.int32)
        f['nclasses'] = np.array([max(LABELS.values())], dtype=np.int32)
        f['dwin'] = np.array([dwin], dtype=np.int32)
        f['start_idx'] = np.array([1], dtype=np.int32)
        f['end_idx'] = np.array([2], dtype=np.int32)

    print "Done."



if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
