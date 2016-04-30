import resource

from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist

import argparse
import numpy as np
import sys

matrix = matlist.blosum62
gap_open = -10
gap_extend = -0.5

def parse_human(data_file, start_idx, count):
    X_strings = []

    ditch_count = 0

    next_line_is_seq = False
    next_line_is_output = False

    with open(data_file, 'r') as f:
        for line in f:
			if len(X_strings) >= count:
				break
			elif 'A:sequence' in line:
			    next_line_is_output = False
			    next_line_is_seq = True
			elif 'A:secstr' in line:
			    next_line_is_seq = False
			    next_line_is_output = True
			elif (':sequence' in line) or ('secstr' in line):
			    next_line_is_seq = False
			    next_line_is_output = False
			elif next_line_is_seq:
				ditch_count += 1
				if ditch_count > start_idx:
					X_strings.append(line[:-1])

    return X_strings

def parse_princeton(data_file, num_proteins):
    X_strings = []

    # NoSeq is padding
    acid_order = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','NoSeq']

    data = np.load(data_file)
    amino_acids = np.reshape(data, (num_proteins, 700, 57))
    
    for p in range(num_proteins):
        x = ''
        for a in range(700):
            # reached end of this sequence
            if amino_acids[p][a][21]:
                # label should also be noseq
                assert amino_acids[p][a][30]
                X_strings.append(x)
                break
            else:
                this_acids = np.nonzero(amino_acids[p][a][:22])[0]
                assert len(this_acids) == 1
                this_acid = acid_order[this_acids[0]]

                x = x+this_acid


    return X_strings

def sim(aln_seq1,aln_seq2):
    c=0.0
    for i,j in zip(aln_seq1,aln_seq2):
        if i ==j:
            c=c+1
    return c/min(len(aln_seq1),len(aln_seq2))

def identity_score(seq1, seq2):
	try:
		top_aln = pairwise2.align.globalds(seq1,seq2, matrix, gap_open, gap_extend, one_alignment_only=1)[0]
		aln_seq1, aln_seq2, score, begin, end = top_aln
		return sim(aln_seq1,aln_seq2)
	except KeyError:
		# force remove sequence
		return 1.0

def mock_score(seq1, seq2):
	return 0.0

def pfilter(sequences, filter_out, start_idx, l, identity_thresh=0.25, identity_map=identity_score, verbose=True):
	seq_idxs = []
	for idx in range(0, l):
		# if idx % 1 == 0:
		# 	print "Filtering sequence ", idx+1, "... (", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6
		seq1 = sequences[idx]
		keep = True
		for seq2 in filter_out:
			if identity_map(seq1, seq2) >= identity_thresh:
				keep = False
				break
		if keep:
			seq_idxs.append(start_idx+idx)
			if verbose:
				print start_idx+idx
	return seq_idxs

def main(arguments):
	global args
	parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument('cb513', help="CB513 path", type=str)
	parser.add_argument('train', help="Training set path", type=str)
	parser.add_argument('start', help="Start index", type=int)
	parser.add_argument('count', help="Number of indices", type=int)

	args = parser.parse_args(arguments)
	cb513_path = args.cb513
	train_path = args.train
	start_idx = args.start
	count = args.count

	train_seqs = parse_human(train_path, start_idx, count)
	cb513_seqs = parse_princeton(cb513_path, 514)

	pfilter(train_seqs, cb513_seqs, start_idx, count)



if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
