#!/usr/bin/env python

"""Language modeling postprocessing
"""
from scipy.spatial.distance import cosine

import numpy as np
import h5py
import argparse
import sys
import re
import heapq


def load_vocab_dict(file_path):
    words_to_idx = {}
    idx_to_words = {}

    with open(file_path, 'r') as f:
        for line in f:
            idx, word, _ = line.strip().split()
            words_to_idx[word] = int(idx)-1
            idx_to_words[int(idx)-1] = word

    return words_to_idx, idx_to_words

def k_nearest(idx, k, embeddings, sim_op):
	all_sims = []
	main_embedding = embeddings[idx]

	for i, embedding in enumerate(embeddings):
		if i != idx:
			this_dist = sim_op(main_embedding, embedding)
			heapq.heappush(all_sims, (this_dist, i, idx))

	return heapq.nlargest(k, all_sims)

def find_top_pairs(embeddings, k, sim_op):
	top_sims = []

	for idx in range(len(embeddings)):
		these_sims = k_nearest(idx, k, embeddings, sim_op)
		top_sims = heapq.merge(top_sims, these_sims)
		# for efficiency
		top_sims = heapq.nlargest(k, top_sims)

	return top_sims


args = {}

def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('datafile', help="Embedding HDF5 file",
                        type=str)
    parser.add_argument('vocab', help="Vocab file", type=str)

    args = parser.parse_args(arguments)
    vocab_file = args.vocab
    embedding_file = args.datafile

    K = 100
    sim = lambda v1, v2: 1-cosine(v1, v2)
    words = ["tuesday", "francisco", "september", "japan"]


    print "Loading vocab from "+vocab_file
    words_to_idx, idx_to_words = load_vocab_dict(vocab_file)


    print "Loading from "+embedding_file
    with h5py.File(embedding_file, "r") as f:
    	embeddings = f['embedding'][:]

    	top_pairs = find_top_pairs(embeddings, K, sim)
    	for tp in top_pairs:
    		idx1 = tp[1]
    		idx2 = tp[2]
    		print idx_to_words[idx1],
    		print idx_to_words[idx2]
    	
    	# for word in words:
    	# 	word_idx = words_to_idx[word]
    	# 	print word
	    # 	nearest_idxs = k_nearest(word_idx, K, embeddings, sim)
	    # 	for i in nearest_idxs:
	    # 		wordidx = i[1]
	    # 		print idx_to_words[wordidx]
	    # 	print


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))