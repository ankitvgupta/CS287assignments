import numpy as np 
import pandas as pd 
import csv 
import sys

pd.set_option('expand_frame_repr', False)

sets = []
accuracies = []

for line in sys.stdin:

	if line[0] == "=" or len(line) == 1:
		continue
	if line[0] == 'A':
		accuracies.append(line.split('\t')[1])
	if line[0] == 'D':
		splitted = line.split('\t')
		sets.append([splitted[1], splitted[3], splitted[5], splitted[7], splitted[9], splitted[11], splitted[13], splitted[15]])

sets = np.array(sets)
accuracies = np.array(accuracies)
alldata = np.append(sets, accuracies[:, np.newaxis], axis=1)

df = pd.DataFrame(alldata)
df.columns = ['Datafile', 'Classifier', 'Alpha', 'Eta', 'Lambda','MinibatchSize','NumEpochs', 'MinimumSentenceLength', 'Accuracy']

print df.sort(columns='Accuracy', ascending=False)



