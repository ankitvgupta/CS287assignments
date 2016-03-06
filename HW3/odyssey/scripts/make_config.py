# Use this to quickly make configurations for experiments
import itertools

datafiles=["SMALL_1.hdf5", "SMALL_2.hdf5", "SMALL_5.hdf5"]
classifier=["nn"]
eta=[.001,.01, 1, 100]
minibatchsize=[32,128,1024]
numepochs=[20]
alpha=[1]
lambdas=[1]
optimizer=["sgd", "adagrad"]
hiddenlayers=[10,50,100]
embeddingsize=[25,50,100]

lists = [datafiles,classifier,eta,minibatchsize,numepochs,alpha,lambdas,optimizer,hiddenlayers,embeddingsize]
print "#datafile, classifier, eta, minibatchsize, numepochs=20, alpha, lambda=0, optimizer, hiddenlayers, embeddingsize"
for element in itertools.product(*lists):
    #print element
    print ' '.join(map(str,element))