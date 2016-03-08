# Use this to quickly make configurations for experiments
import itertools
print_description = True
datafiles=["PTB_1.hdf5", "PTB_2.hdf5", "PTB_3.hdf5", "PTB_4.hdf5", "PTB_5.hdf5"]
classifier=["nce"]
eta=[1, 2, .1, .5, 10, 50, .01]
minibatchsize=[128]
numepochs=[20, 40]
alpha=[.0001, 1]
lambdas=[1]
optimizer=["sgd"]
hiddenlayers=[50, 100]
embeddingsize=[50]
K=[5,20,40]

lists = [datafiles,classifier,eta,minibatchsize,numepochs,alpha,lambdas,optimizer,hiddenlayers,embeddingsize,K]
if not print_description:
    print "#datafile, classifier, eta, minibatchsize, numepochs=20, alpha, lambda=0, optimizer, hiddenlayers, embeddingsize,K"
    for element in itertools.product(*lists):
        #print element
        print ' '.join(map(str,element))
else:
    print lists

