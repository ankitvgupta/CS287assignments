# Use this to quickly make configurations for experiments
import itertools
print_description = False
datafiles=["PTB.hdf5"]
classifier=["neural"]
alpha=[1]
beta=[]
embedding_size=[15,25,45,55]
minibatch_size=[320]
optimizer=["sgd","adagrad"]
epochs=[100]
hidden=[15,25]
eta=[.001, .01, .1, .0001]


lists = [datafiles,classifier,alpha, beta, embedding_size, minibatch_size, optimizer, epochs, hidden, eta]
if not print_description:
    print "#datafile, classifier, alpha, beta, embedding_size, minibatch_size, optimizer, epoch, hidden, eta"
    for element in itertools.product(*lists):
        print ' '.join(map(str,element))
else:
    print lists

