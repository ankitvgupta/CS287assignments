# Use this to quickly make configurations for experiments
import itertools
print_description = False
datafiles=["PTB.hdf5"]
classifier=["neural"]
window_size=[1,2,3]
b=[32, 128]
alpha=[1]
sequence_length=[10]
embedding_size=[15,25,45,55]
optimizer=["sgd","adagrad"]
epochs=[100]
hidden=[15,25]
eta=[.001, .01, .1, .0001]
rnn1=["lstm"]
rnn2=["none"]
dropout=[.5]

lists = [datafiles,classifier,window_size, b, alpha, sequence_length, embedding_size, optimizer, epochs, hidden, eta, rnn1, rnn2, dropout]
if not print_description:
    print "#datafile, classifier, window_size, b, alpha, sequence_length, embedding_size, optimizer, epochs, hidden, eta, rnn1, rnn2, dropout"
    for element in itertools.product(*lists):
        print ' '.join(map(str,element))
else:
    print lists

