#!/usr/bin/env python

"""Text Classification Exploratory Analysis
"""
from collections import defaultdict
from matplotlib import pyplot as plt

import argparse
import h5py
import numpy as np
import preprocess as pp
import sys


def plot_word_count_distribution(sentences, outfile=None):
	word_to_count = word_count(sentences)
	x = np.log(word_to_count.values())
	fig = plt.figure()
	n, bins, patches = plt.hist(x, 100, normed=1, facecolor='green', alpha=0.75)
	plt.xlabel("Word Count")
	plt.ylabel("Log(Count Frequency)")
	plt.title("Word Count Distribution (Log scale)")
	if outfile is not None:
		fig.savefig(outfile)
		print "Saved word count distribution plot to "+outfile
	else:
		plt.show()


def word_count(sentences):
	word_to_count = defaultdict(int)
	for sentence in sentences:
		for word in sentence:
			word_to_count[word] += 1
	return word_to_count

def top_n_words(sentences, N):
	word_to_count = word_count(sentences)
	largest_keys = sorted(word_to_count, key=word_to_count.get, reverse=True)[:N]
	largest_values = [word_to_count[k] for k in largest_keys]
	return dict(zip(largest_keys, largest_values))

def print_word_dict(word_dict, value_title="COUNT"):
	words = sorted(word_dict, key=word_dict.get, reverse=True)
	print "WORD:",
	print value_title
	for word in words:
		print word+":",
		print word_dict[word]


def file_list_to_sentences(file_list, dataset):
	sentences = []
	for filename in file_list:
		if filename:
		    with open(filename, "r") as f:
		        for line in f:
		            words = pp.line_to_words(line, dataset)
		            sentences.append(words)
	return sentences

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set",
                        type=str)
    args = parser.parse_args(arguments)
    dataset = args.dataset

    train, valid, test = pp.FILE_PATHS[dataset]
    sentences = file_list_to_sentences([train, valid, test], dataset)

    N = 10
    top_words = top_n_words(sentences, N)
    print_word_dict(top_words)

    plot_word_count_distribution(sentences, 'wordcountdist.png')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
