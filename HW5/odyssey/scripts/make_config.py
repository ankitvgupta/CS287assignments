# Use this to quickly make configurations for experiments
import itertools
print_description = True
datafiles=['CONLL_3.hdf5']
classifier=['struct']
alpha=[1]
beta=[1]
embedding_size=[25, 50, 75, 100]
minibatch_size=[1]
optimizer=["sgd"]
epochs=[10, 30, 50]
hidden=[0, 25, 35, 55]
eta=[0.01, 0.005, 0.001, 0.0005, .1]


lists = [datafiles,classifier,alpha, beta, embedding_size, minibatch_size, optimizer, epochs, hidden, eta]
if not print_description:
    print "#datafile, classifier, alpha, beta, embedding_size, minibatch_size, optimizer, epoch, hidden, eta"
    for element in itertools.product(*lists):
        print ' '.join(map(str,element))
else:
    print lists

