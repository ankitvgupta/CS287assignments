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


def plot_gram_count_distribution(sentences, grams=1, outfile=None):
	gram_dict = gram_count(sentences, grams)
	gram_name = str(grams)+"-Gram"
	x = np.log(gram_dict.values())
	fig = plt.figure()
	n, bins, patches = plt.hist(x, 100, normed=1, facecolor='green', alpha=0.75)
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
		print gram+":",
		print gram_dict[gram]

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
    for i in range(1, 4):
    	print "Top "+str(N)+" "+str(i)+"-grams"
    	top_grams = top_n_grams(sentences, N, i)
    	print_gram_dict(top_grams)
    	print
    	plot_gram_count_distribution(sentences, i, str(i)+'countdist.png')

    #plot_word_count_distribution(sentences, 'wordcountdist.png')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
