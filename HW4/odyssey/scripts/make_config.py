# Use this to quickly make configurations for experiments
import itertools
print_description = False
datafiles=["PTB.hdf5"]
classifier=["rnn"]
window_size=[2]
b=[32, 64, 128]
alpha=[1]
sequence_length=[10, 20,  32, 64, 128]
embedding_size=[15,25,35,45,55]
optimizer=["sgd","adagrad"]
epochs=[100]
hidden=[100]
eta=[.001, .0001, .01, .005]

lists = [datafiles,classifier,window_size, b, alpha, sequence_length, embedding_size, optimizer, epochs, hidden, eta]
if not print_description:
    print "#datafile, classifier, window_size, b, alpha, sequence_length, embedding_size, optimizer, epochs, hidden, eta"
    for element in itertools.product(*lists):
        print ' '.join(map(str,element))
else:
    print lists
