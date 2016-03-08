# Use this to quickly make configurations for experiments
import itertools
print_description = True
datafiles=["SMALL_1.hdf5", "SMALL_2.hdf5", "SMALL_3.hdf5", "SMALL_4.hdf5", "SMALL_5.hdf5", "PTB_1.hdf5", "PTB_2.hdf5", "PTB_3.hdf5", "PTB_4.hdf5", "PTB_5.hdf5"]
classifier=["multinomial"]
eta=[1]
minibatchsize=[1]
numepochs=[20]
alpha=[.001, .0001, .01, .1, 1, 2, 4, 10]
lambdas=[1]
optimizer=["sgd"]
hiddenlayers=[50]
embeddingsize=[50]
K=[5]

lists = [datafiles,classifier,eta,minibatchsize,numepochs,alpha,lambdas,optimizer,hiddenlayers,embeddingsize,K]
if not print_description:
    print "#datafile, classifier, eta, minibatchsize, numepochs=20, alpha, lambda=0, optimizer, hiddenlayers, embeddingsize,K"
    for element in itertools.product(*lists):
        #print element
        print ' '.join(map(str,element))
else:
    print lists

