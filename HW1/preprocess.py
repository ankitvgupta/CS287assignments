#!/usr/bin/env python

"""Text Classification Preprocessing
"""
from nltk.corpus import stopwords

import features
import numpy as np
import h5py
import argparse
import sys
import re

STOP_WORDS = set(stopwords.words('english'))


def line_to_words(line, dataset, exclude_stops=False, exclude_aposts=False):
    # Different preprocessing is used for these datasets.
    if dataset not in ['SST1', 'SST2']:
        clean_line = clean_str_sst(line.strip())
    else:
        clean_line = clean_str(line.strip())
    clean_line = clean_str_sst(line.strip())
    words = clean_line.split(' ')
    words = words[1:]
    if exclude_stops:
        words = [w for w in words if w not in STOP_WORDS]
    if exclude_aposts:
        words = [w for w in words if "'" not in w]
    return words

def pad_sentence(sent, max_features, padder=1):
    if len(sent) < max_features:
        sent.extend([padder] * (max_features - len(sent)))

def prepare_features(data_name, feature_list, dataset):
    sentences = []
    inited_features = []
    with open(data_name, 'r') as f:
        for line in f:
            sentence = line_to_words(line, dataset)
            sentences.append(sentence)

    index_offset = 2
    max_feat_length = 0
    for feature in feature_list:

        if type(feature) is tuple:
            kwargs = feature[1]
            feature = feature[0]
        else:
            kwargs = {}

        inited_feature = feature(index_offset=index_offset, **kwargs)
        inited_feature.initialize(sentences)
        inited_features.append(inited_feature)
        index_offset += inited_feature.totalFeatureCount()
        max_feat_length += inited_feature.maxFeatureLength()

    return inited_features, max_feat_length, index_offset

def convert_data(data_name, feature_list, max_features, dataset):
    features = []
    lbl = []

    with open(data_name, 'r') as f:
        for line in f:

            y = int(line[0]) + 1
            lbl.append(y)

            sentence = line_to_words(line, dataset)
            all_features = [feat.sentenceToFeatures(sentence) for feat in feature_list]
            sentence_features = reduce(lambda l1, l2: l1+l2, all_features, [])
            pad_sentence(sentence_features, max_features)
            features.append(sentence_features)

    return np.array(features, dtype=np.int32), np.array(lbl, dtype=np.int32)

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# Different data sets to try.
# Note: TREC has no development set.
# And SUBJ and MPQA have no splits (must use cross-validation)
FILE_PATHS = {"SST1": ("data/stsa.fine.phrases.train",
                       "data/stsa.fine.dev",
                       "data/stsa.fine.test"),
              "SST2": ("data/stsa.binary.phrases.train",
                       "data/stsa.binary.dev",
                       "data/stsa.binary.test"),
              "TREC": ("data/TREC.train.all", None,
                       "data/TREC.test.all"),
              "SUBJ": ("data/subj.all", None, None),
              "MPQA": ("data/mpqa.all", None, None)}
args = {}


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set",
                        type=str)
    args = parser.parse_args(arguments)
    dataset = args.dataset
    train, valid, test = FILE_PATHS[dataset]

    feature_list = [(features.NgramFeature, {'N': 1}), features.SentimentFeature]
    prepared_features, max_features, total_features = prepare_features(train, feature_list, dataset)
    train_input, train_output = convert_data(train, prepared_features, max_features, dataset)

    if valid:
        valid_input, valid_output = convert_data(valid, prepared_features, max_features, dataset)
    if test:
        test_input, _ = convert_data(test, prepared_features, max_features, dataset)

    V = total_features-2

    print "Loaded "+str(V)+" features."
    C = np.max(train_output)

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
