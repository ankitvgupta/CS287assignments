# Use this to quickly make configurations for experiments
import itertools
print_description = True
datafiles=["CONLL_1.hdf5", "CONLL_3.hdf5"]
classifier=["memm"]
alpha=[1]
beta=[1]
embedding_size=[15, 25, 50, 75, 100]
minibatch_size=[64, 128, 256]
optimizer=["sgd", "adagrad"]
epochs=[50]
hidden=[25, 35, 55]
eta=[.01, .1, 1]

lists = [datafiles,classifier,alpha, beta, embedding_size, minibatch_size, optimizer, epochs, hidden, eta]
if not print_description:
    print "#datafile, classifier, alpha, beta, embedding_size, minibatch_size, optimizer, epoch, hidden, eta"
    for element in itertools.product(*lists):
        print ' '.join(map(str,element))
else:
    print lists

