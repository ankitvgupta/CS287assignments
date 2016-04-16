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

def load_tag_dict(file_path):
    tag_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            tag, idx = line.strip().split(' ')
            tag_dict[tag] = int(idx)

    return tag_dict

def parse_line(s, num_sub="NUMBER", lowercase=False):
    assert s[0].isdigit()
    s = s.strip()
    global_id, sentence_id, word, tag = s.split('\t')
    if lowercase:
        word = word.lower()
    word = re.sub('\d', num_sub, word)
    return word, tag

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

def plot_tag_distribution(tag_names, tag_counts, outfile=None):
	sns.set_context("paper")

	fig = plt.figure()
	x_pos = np.arange(len(tag_names))

	plt.bar(x_pos, tag_counts, facecolor='green', alpha=0.75)

	locs, labels = plt.xticks(x_pos, tag_names)
	plt.setp(labels, rotation=90, ha='left')
	plt.ylabel("Count")
	plt.title("Class Count Distribution")
	if outfile is not None:
		fig.savefig(outfile)
		print "Saved POS count distribution plot to "+outfile
	else:
		plt.show()

def plot_rare_counts(rank_list, outfile=None):

	x = range(0, 101)
	y = []
	for p in x:
		y.append(count_rare_words(p, rank_list))

	sns.set_context("paper")

	fig = plt.figure()
	plt.plot(x, y)

	plt.xlabel("Percent Vocab Included")
	plt.ylabel("Number Rare Words")
	plt.title("Rare Word Frequency vs Vocabulary Size")
	if outfile is not None:
		fig.savefig(outfile)
		print "Saved rare words count distribution plot to "+outfile
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

def vocab_file_to_percent_dict(vocab_file):
	word_counts = defaultdict(int)
	count = 0

	with open(vocab_file, 'rb') as f:
		for line in f:
			count += 1
			split_line = line.split()
			word = split_line.pop(0)
			clean_word = re.sub('\d', "NUMBER", word).lower()
			word_counts[clean_word] = count

	for word, v in word_counts.items():
		word_counts[word] = v/(1.0*count)

	return word_counts

def test_set_to_rank_list(test_file, percent_dict):
	rank_list = []
	with open(test_file, 'r') as f:
	    for line in f:
	        if line[0].isdigit():
	            word, _ = parse_line(line)
	            clean_word = re.sub('\d', "NUMBER", word).lower()
	            rank_list.append(percent_dict[clean_word])
	return rank_list

def count_rare_words(p, rank_list):
	c = 0
	percent = p*0.01
	for rank in rank_list:
		if percent < rank:
			c += 1
	return c

def file_list_to_tag_count(file_list, tag_dict):
	tag_count = defaultdict(int)

	for filename in file_list:
		if filename:
			with open(filename, 'r') as f:
			    for line in f:
			        if line[0].isdigit():
			            _, tag = parse_line(line)
			            tag_count[tag_dict[tag]] += 1
	
	tag_names = []
	tag_counts = []
	for tag in tag_dict:
		tag_names.append(tag)
		tag_counts.append(tag_count[tag_dict[tag]])

	return tag_names, tag_counts

def file_list_to_sentences(file_list):
	sentences = []
	for filename in file_list:
		if filename:
			this_sentence = []
			with open(filename, 'r') as f:
			    for line in f:
			        if line[0].isdigit():
			            word, _ = parse_line(line)
			            this_sentence.append(word)
			        # NEW SENTENCE!
			        else:
			            # finish the old
			            sentences.append(this_sentence)
			            this_sentence = []

			sentences.append(this_sentence)
	
	return sentences

FILE_PATHS = {"CONLL": ("data/train.num.txt",
                        "data/dev.num.txt",
                        "data/test.num.txt",
                        "data/tags.txt")}


args = {}


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set",
                        type=str)
    args = parser.parse_args(arguments)
    dataset = args.dataset

    train, valid, test, tag_file = FILE_PATHS[dataset]
    sentences = file_list_to_sentences([train, valid])
    tag_dict = load_tag_dict(tag_file)

    tag_names, tag_counts = file_list_to_tag_count([train, valid], tag_dict)
    plot_tag_distribution(tag_names, tag_counts, 'tagcounts.png')

    # perc_dict = vocab_file_to_percent_dict('data/glove.6B.50d.txt')
    # rank_list = test_set_to_rank_list(test, perc_dict)
    # plot_rare_counts(rank_list, 'rarecounts.png')

    N = 10
    for i in range(1, 4):
    	print "Top "+str(N)+" "+str(i)+"-grams"
    	top_grams = top_n_grams(sentences, N, i)
    	print_gram_dict(top_grams)
    	print
    	plot_gram_count_distribution(sentences, i, str(i)+'countdist.png')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
