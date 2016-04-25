# Use this to quickly make configurations for experiments
import itertools
print_description = True
datafiles=["PRINC_CB513_1.hdf5"]
classifier=["rnn"]
b=[128]
alpha=[1]
sequence_length=[50, 100, 200]
embedding_size=[100,200, 300, 500]
optimizer=["sgd","adagrad"]
epochs=[200]
hidden=[100, 200]
eta=[.001, .01, .1, .05]
rnn1=["lstm"]
rnn2=["lstm"]
dropout=[.5]

lists = [datafiles,classifier, b, alpha, sequence_length, embedding_size, optimizer, epochs, hidden, eta, rnn1, rnn2, dropout]
if not print_description:
    print "#datafile, classifier, b, alpha, sequence_length, embedding_size, optimizer, epochs, hidden, eta, rnn1, rnn2, dropout"
    for element in itertools.product(*lists):
        print ' '.join(map(str,element))
else:
    print lists

