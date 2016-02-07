"""Text Classification Postprocessing
"""
from features import SentimentFeature

import argparse
import sys

def print_class_confusion(f):
	class_confusion = dict([(i, [0 for _ in range(3)]) for i in range(1, 6)])
	for line in f:
		idx, y, yp = line.split(',')
		if int(y) == int(yp):
			class_confusion[int(y)][0] += 1
		else:
			class_confusion[int(yp)][1] += 1
			class_confusion[int(y)][2] += 1

	for c in class_confusion:
		print "Class",
		print c
		print "# Correct and Guessed:",
		print class_confusion[c][0]
		print "# Incorrect and Guessed",
		print class_confusion[c][1]
		print "# Correct and Not Guessed",
		print class_confusion[c][2]
		print "# Total",
		print class_confusion[c][0] + class_confusion[c][2]
		print

def print_class_tradeoffs(f):
	class_tradeoffs = {}
	for c1 in range(1, 6):
		class_tradeoffs[c1] = {}
		for c2 in range(1, 6):
			class_tradeoffs[c1][c2] = 0

	for line in f:
		_, y, yp = line.split(',')
		y = int(y)
		yp = int(yp)

		class_tradeoffs[y][yp] += 1

	for c1 in range(1, 6):
		print
		for c2 in range(1, 6):
			print class_tradeoffs[c1][c2],

	return 


def print_sentences(f, vf, actual_class, pred_class, count=1, dataset='SST1'):
	sf = SentimentFeature()
	idxs = []
	for line in f:
		idx, y, yp = line.split(',')
		if int(y) == actual_class and int(yp) == pred_class:
			idxs.append(int(idx))

	i = 1
	c = 0
	for line in enumerate(vf):
		_, sentence = line
		if i in idxs:
			print sentence[2:]
			c += 1
			if c >= count:
				break
		i += 1


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('valid_file', help="Valid file",
                        type=str)

    args = parser.parse_args(arguments)
    valid_file = args.valid_file
    valid_data = "data/stsa.fine.dev"


    for i in range(1, 6):
    	for j in range(1, 6):
			print i,
			print j
			with open(valid_file) as f:
				#print_class_tradeoffs(f)
				with open(valid_data) as vf:
					print_sentences(f, vf, i, j)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))