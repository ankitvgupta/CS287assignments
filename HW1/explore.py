#!/usr/bin/env python

"""Text Classification Exploratory Analysis
"""
from collections import defaultdict
from matplotlib import pyplot as plt
from textblob import TextBlob

import argparse
import h5py
import numpy as np
import preprocess as pp
import sys

def plot_sentiment_distribution(sentences, labels, outfile=None):
	counts = {}
	label_dict = {}
	cm_dict = {}
	counts['polarity'] = []
	label_dict['polarity'] = []
	cm_dict['polarity'] = plt.cm.get_cmap('bwr')
	counts['subjectivity'] = []
	label_dict['subjectivity'] = []
	cm_dict['subjectivity'] = plt.cm.get_cmap('PRGn')

	for idx, sentence in enumerate(sentences):
		blob = TextBlob(' '.join(sentence))
		polarity = (blob.sentiment.polarity+1.0)/2
		subjectivity = blob.sentiment.subjectivity
		if polarity != 0.5:
			counts['polarity'].append(polarity)
			label_dict['polarity'].append(labels[idx])
		if subjectivity != 0.0:
			counts['subjectivity'].append(subjectivity)
			label_dict['subjectivity'].append(labels[idx])

	for t in ['polarity', 'subjectivity']:

		cm = cm_dict[t]

		fig = plt.figure()
		n, bins, patches = plt.hist(counts[t], 100, facecolor='blue', alpha=0.75)
		
		bin_cols = []
		for i in range(len(bins)-1):

			start = bins[i]
			end = bins[i+1]

			all_classes_in_bin = []

			for j, count in enumerate(counts[t]):
				if count >= start and count <= end:
					if t == 'subjectivity':
						lab_val = 0.5*(abs(label_dict[t][j]-2))
					else:
						lab_val = 0.25*(label_dict[t][j])
					all_classes_in_bin.append(lab_val)

			l = len(all_classes_in_bin)
			if l:
				bin_col = sum(all_classes_in_bin)/(1.0*l)
			else:
				bin_col = 0.5

			bin_cols.append(bin_col)


		for c, p in zip(bin_cols, patches):
			plt.setp(p, 'facecolor', cm(c))

		plt.xlabel(t.capitalize())
		plt.ylabel("Count")
		plt.title("SST1 training set "+t+" distribution")
		if outfile is not None:
			fn = outfile+t+'.png'
			fig.savefig(fn)
			print "Saved "+t+" histogram to "+fn
		else:
			plt.show()	

def plot_gram_count_distribution(sentences, grams=1, outfile=None):
	gram_dict = gram_count(sentences, grams)
	gram_name = str(grams)+"-Gram"
	x = np.log(gram_dict.values())
	fig = plt.figure()

	n, bins, patches = plt.hist(x, 100/grams, facecolor='green', alpha=0.75)

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

def file_list_to_sentences(file_list, dataset):
	sentences = []
	labels = []
	for filename in file_list:
		if filename:
		    with open(filename, "r") as f:
		        for line in f:
		            words = pp.line_to_words(line, dataset)
		            sentences.append(words)
		            labels.append(int(line[0]))
	return sentences, labels

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set",
                        type=str)
    args = parser.parse_args(arguments)
    dataset = args.dataset

    train, valid, test = pp.FILE_PATHS[dataset]
    sentences, labels = file_list_to_sentences([train, valid, test], dataset)

    N = 10
    for i in range(1, 4):
    	print "Top "+str(N)+" "+str(i)+"-grams"
    	top_grams = top_n_grams(sentences, N, i)
    	print_gram_dict(top_grams)
    	print
    	plot_gram_count_distribution(sentences, i, str(i)+'countdist.png')

    plot_sentiment_distribution(sentences, labels, 'exploratory_results/sentiment')

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
