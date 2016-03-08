#!/usr/bin/env python

"""POS Exploratory Analysis
"""
from collections import defaultdict
from matplotlib import pyplot as plt
from textblob import TextBlob

import argparse
import h5py
import numpy as np
import re
import seaborn as sns
import sys

def load_vocab_dict(file_path):
    vocab_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            idx, word, _ = line.strip().split()
            vocab_dict[word] = int(idx)

    return vocab_dict

def parse_line(s, vocab):
    s = s.lower().strip()

    idxs = []

    for word in s.split():
        if word not in vocab:
            word = '<unk>'
        idxs.append(vocab[word])

    return idxs

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

def num_words_per_context(Xs, Ys):
	context_dict = {} # contexts to word sets

	for i, x in enumerate(Xs):
		w = Ys[i]
		c = ';'.join([str(j) for j in x])
		if c in context_dict:
			context_dict[c].add(w)
		else:
			context_dict[c] = set([w])

	return [len(v) for v in context_dict.values()]

def plot_words_per_context(x, context_size=1, outfile=None):
	sns.set_context("paper")

	fig = plt.figure()

	n, bins, patches = plt.hist(np.log(x), facecolor='green')

	plt.xlabel("Log(Unique Words Per Context)")
	plt.ylabel("Count")
	plt.title("Number of Unique Words Per Context with "+str(context_size)+"-gram contexts")
	if outfile is not None:
		fig.savefig(outfile)
		print "Savedplot to "+outfile
	else:
		plt.show()

def plot_gram_count_distribution(sentences, grams=1, outfile=None):
	sns.set_context("paper")

	gram_dict = gram_count(sentences, grams)
	gram_name = str(grams)+"-Gram"
	x = np.log(gram_dict.values())
	fig = plt.figure()

	n, bins, patches = plt.hist(x, 30/grams, facecolor='green', alpha=0.75)

	plt.xlabel("Log("+gram_name+" Count)")
	plt.ylabel("Count Frequency")
	plt.title(gram_name+" Count Distribution (Log scale)")
	if outfile is not None:
		fig.savefig(outfile)
		print "Saved gram count distribution plot to "+outfile
	else:
		plt.show()

def gram_count(sentences, grams=1):
	count = defaultdict(int)
	for sentence in sentences:
		for start_idx in range(0, len(sentence)-(grams-1)):
			gram = ' '.join(sentence[start_idx:start_idx+grams])
			count[gram] += 1
	print len(count)
	return count

def top_n_grams(sentences, N, grams=1):
	gram_dict = gram_count(sentences, grams)
	largest_keys = sorted(gram_dict, key=gram_dict.get, reverse=True)[:N]
	largest_values = [gram_dict[k] for k in largest_keys]
	return dict(zip(largest_keys, largest_values))

def print_gram_dict(gram_dict, value_title="COUNT"):
	grams = sorted(gram_dict, key=gram_dict.get, reverse=True)
	print "GRAM:",
	print value_title
	for gram in grams:
		print gram,
		print "("+str(gram_dict[gram])+")"

def create_training_data(data_file, vocab, dwin):
    Xs = []
    Ys = []

    with open(data_file, 'rb') as f:
        for line in f:
            sentence = parse_line(line, vocab)
            X, Y = split_idxs(sentence, dwin)
            Xs.extend(X)
            Ys.extend(Y)

    return Xs, Ys

FILE_PATHS = {"PTB": ("data/train.txt",
                      "data/words.dict"),
              "SMALL": ("data/train.1000.txt",
                       "data/words.1000.dict")}
args = {}


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set",
                        type=str)
    args = parser.parse_args(arguments)
    dataset = args.dataset

    for dwin in range(1, 6):

	    train, word_dict = FILE_PATHS[dataset]
	    vocab_dict = load_vocab_dict(word_dict)
	    train_input, train_output = create_training_data(train, vocab_dict, dwin)

	    x = num_words_per_context(train_input, train_output)
	    filename = 'words_per_context_'+str(dwin)+'.png'
	    plot_words_per_context(x, dwin, outfile=filename)

    # N = 10
    # for i in range(1, 4):
    # 	print "Top "+str(N)+" "+str(i)+"-grams"
    # 	top_grams = top_n_grams(sentences, N, i)
    # 	print_gram_dict(top_grams)
    # 	print
    # 	plot_gram_count_distribution(sentences, i, str(i)+'countdist.png')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
