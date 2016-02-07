# USAGE: python analysis.py {all, SST1, SST2, TREC...} < SUMMARY_FILE
#   Ex: python analysis.py all < SUMMARY_FILE
#       python analysis.py TREC < SUMMARY_FILE
# To create summary file, do tail -n 2 * > SUMMARY_FILE in the directory being summarized
import numpy as np 
import pandas as pd 
import csv 
import sys

pd.set_option('expand_frame_repr', False)

sets = []
accuracies = []
assert(len(sys.argv) == 2)
print(sys.argv)


for line in sys.stdin:

	if line[0] == "=" or len(line) == 1:
		continue
	if line[0] == 'A':
		accuracies.append(line.split('\t')[1])
	if line[0] == 'D':
		splitted = line.split('\t')
		sets.append([splitted[1].split("HW1/")[1], splitted[3], splitted[5], splitted[7], splitted[9], splitted[11], splitted[13], splitted[15]])

sets = np.array(sets)
accuracies = np.array(accuracies)
alldata = np.append(sets, accuracies[:, np.newaxis], axis=1)

df = pd.DataFrame(alldata)
df.columns = ['Datafile', 'Classifier', 'Alpha', 'Eta', 'Lambda','MinibatchSize','NumEpochs', 'MinimumSentenceLength', 'Accuracy']

df.sort(columns='Accuracy', ascending=False, inplace=True)
if sys.argv[1] != "all":
    df = df[df['Datafile'].str.startswith(sys.argv[1])]

print df
print ""
print "Best nb ones"
print df[df['Classifier'] == 'nb']

print ""
print "Best lr ones"
print df[df['Classifier'] == 'lr']

print ""
print "Best hinge ones"
print df[df['Classifier'] == 'hinge']


