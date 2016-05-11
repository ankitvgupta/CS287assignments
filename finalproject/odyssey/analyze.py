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


moreoutput=True


for line in sys.stdin:
    if line[0] == "=" or len(line) == 1 or line[0] == "W":
        continue
    if line[0] == 'A' and line[1] == "c":
        splitted = line.split('\t')
        accuracies.append(map(float, splitted[1:-1]))
	#accuracies.append([float(splitted[1]), float(splitted[2]), float(splitted[3]), float(splitted[4])])
    if line[0] == 'd':
        splitted = line.split('\t')
        #sets.append([splitted[1].split("HW4/")[1], splitted[3], splitted[5], splitted[7], splitted[9], splitted[11], splitted[13], splitted[15], splitted[17], splitted[19], splitted[21], splitted[23]])
        if len(splitted) > 27:
            sets.append([splitted[1].split("finalproject/")[1], splitted[3], splitted[5], splitted[7], splitted[9], splitted[11], splitted[13], splitted[15], splitted[17], splitted[19], splitted[21], splitted[23], splitted[25], splitted[27]])
        else:
            sets.append([splitted[1].split("finalproject/")[1], splitted[3], splitted[5], splitted[7], splitted[9], splitted[11], splitted[13], splitted[15], splitted[17], splitted[19], splitted[21], splitted[23], splitted[25]])

print len(accuracies), len(sets)

#Datafile:       /n/home09/ankitgupta/CS287/CS287assignments/HW2/PTB.hdf5        Classifier:     lr      Alpha:  0       Eta:    50      Lambda: 0.1     Minibatch size: 2000    Num Epochs:     20      Optimizer:      sgd     Hidden Layers:  10      Embedding size: 50
#0.65331391114348

sets = np.array(sets)
accuracies = np.array(accuracies)
print sets.shape, accuracies.shape

alldata = np.append(sets, accuracies, axis=1)



df = pd.DataFrame(alldata)
if sets.shape[1] == 14:
    df.columns = ['Datafile', 'Classifier', 'b', 'alpha', 'sequence_length', 'embedding_size', 'optimizer', 'epochs', 'hidden', 'eta', "RNN1", "RNN2", "Dropout", "Numbidirlayers", 'Accuracy']
elif sets.shape[1] == 13:
    df.columns = ['Datafile', 'Classifier', 'b', 'alpha', 'sequence_length', 'embedding_size', 'optimizer', 'epochs', 'hidden', 'eta', "RNN1", "RNN2", "Dropout", 'Accuracy']
df[['Accuracy']] = df[['Accuracy']].astype(float)

if sys.argv[1] != "all":
    df = df[df['Datafile'].str.startswith(sys.argv[1])]

df.sort(columns='Accuracy', ascending=False, inplace=True)
print "Full, sorted by Accuracy"
print df

df = df[df['RNN1'] == 'lstm']
df = df[df['RNN2'] == 'none']

print "Single LSTM, sorted by Accuracy"
print df


