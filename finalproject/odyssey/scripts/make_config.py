# Use this to quickly make configurations for experiments
import itertools
print_description = False
datafiles=["EPRINC_1.hdf5"]
classifier=["rnn"]
b=[128]
alpha=[1]
sequence_length=[100]
embedding_size=[50, 100]
optimizer=["sgd"]
epochs=[200]
hidden=[100]
eta=[.1, .05, .01]
rnn1=["lstm"]
rnn2=["lstm"]
dropout=[.15, .25, .1, .05] 
layers=[2,3]

lists = [datafiles,classifier, b, alpha, sequence_length, embedding_size, optimizer, epochs, hidden, eta, rnn1, rnn2, dropout, layers]
if not print_description:
    print "#datafile, classifier, b, alpha, sequence_length, embedding_size, optimizer, epochs, hidden, eta, rnn1, rnn2, dropout, layers"
    for element in itertools.product(*lists):
        print ' '.join(map(str,element))
else:
    print lists

