# Use this to quickly make configurations for experiments
import itertools
print_description = True
datafiles=["SMALL_1.hdf5", "SMALL_2.hdf5", "SMALL_5.hdf5", "PTB_1.hdf5", "PTB_2.hdf5", "PTB_5.hdf5"]
classifier=["nce"]
eta=[.001,.01, .0001, .1, 1, 10, 100]
minibatchsize=[256,512,1024]
numepochs=[20]
alpha=[.001]
lambdas=[1]
optimizer=["sgd"]
hiddenlayers=[50,100]
embeddingsize=[50,100]

lists = [datafiles,classifier,eta,minibatchsize,numepochs,alpha,lambdas,optimizer,hiddenlayers,embeddingsize]
if not print_description:
    print "#datafile, classifier, eta, minibatchsize, numepochs=20, alpha, lambda=0, optimizer, hiddenlayers, embeddingsize"
    for element in itertools.product(*lists):
        #print element
        print ' '.join(map(str,element))
else:
    print lists

