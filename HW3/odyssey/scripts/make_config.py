# Use this to quickly make configurations for experiments
import itertools
print_description = False
datafiles=["SMALL_1.hdf5", "SMALL_2.hdf5", "PTB_1.hdf5", "PTB_2.hdf5", "PTB_5.hdf5"]
classifier=["nce"]
eta=[.001, .0001, .1, 1, 10, 100]
minibatchsize=[512,1024]
numepochs=[20]
alpha=[.001]
lambdas=[1]
optimizer=["sgd"]
hiddenlayers=[50,100]
embeddingsize=[50,100]
K=[5,15,25]

lists = [datafiles,classifier,eta,minibatchsize,numepochs,alpha,lambdas,optimizer,hiddenlayers,embeddingsize,K]
if not print_description:
    print "#datafile, classifier, eta, minibatchsize, numepochs=20, alpha, lambda=0, optimizer, hiddenlayers, embeddingsize,K"
    for element in itertools.product(*lists):
        #print element
        print ' '.join(map(str,element))
else:
    print lists

