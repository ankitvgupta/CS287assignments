# Use this to quickly make configurations for experiments
import itertools
print_description = True
datafiles=["PTB.hdf5"]
classifier=["rnn"]
window_size=[2]
b=[32, 64]
alpha=[1]
sequence_length=[16, 32]
embedding_size=[35,45,55]
optimizer=["sgd","adagrad"]
epochs=[50]
hidden=[15]
eta=[.001, .01, .1]
rnn1=["gru", "lstm"]
rnn2=["gru", "lstm", "none"]
dropout=[.5, .25]

lists = [datafiles,classifier,window_size, b, alpha, sequence_length, embedding_size, optimizer, epochs, hidden, eta, rnn1, rnn2, dropout]
if not print_description:
    print "#datafile, classifier, window_size, b, alpha, sequence_length, embedding_size, optimizer, epochs, hidden, eta, rnn1, rnn2, dropout"
    for element in itertools.product(*lists):
        print ' '.join(map(str,element))
else:
    print lists

