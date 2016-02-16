-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

-- For local use, use these.
dofile('nb.lua')
dofile('logisticregression2.lua')

-- For Odyssey, uncomment these
--dofile('/n/home09/ankitgupta/CS287/CS287assignments/HW2/nb.lua')
--dofile('/n/home09/ankitgupta/CS287/CS287assignments/HW2/logisticregression2.lua')


cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'lr', 'classifier to use')
cmd:option('-alpha', 1, 'laplacian smoothing factor for NB')
cmd:option('-eta', 50, 'Learning rate (.1 for adagrad, 500 for sgd, 10 for nn sgd)')
cmd:option('-lambda', 0, 'regularization penalty (.0001 seems to work well for sgd)')
cmd:option('-minibatch', 2000, 'Minibatch size (500 for nn, 2000 for lr)')
cmd:option('-epochs', 20, 'Number of epochs of SGD')
cmd:option('-optimizer', 'adagrad', 'Name of optimizer to use (not yet implemented)')
cmd:option('-hiddenlayers', 10, 'Number of hidden layers (if using neural net)')
cmd:option('-lossfunction', "nll", 'not implemented')
-- Hyperparameters
-- ...


function main() 
   -- Parse input params
   opt = cmd:parse(arg)
   printoptions(opt)
   local f = hdf5.open(opt.datafile, 'r')

   local nclasses = f:read('numClasses'):all():long()[1]
   local nsparsefeatures = f:read('numSparseFeatures'):all():long()[1]
   local ndensefeatures = f:read('numDenseFeatures'):all():long()[1]

   print("nclasses:", nclasses, "nsparsefeatures:", nsparsefeatures, "ndensefeatures:", ndensefeatures)

   local sparse_training_input = f:read('train_sparse_input'):all():long()
   local dense_training_input = f:read('train_dense_input'):all():double()
   local training_output = f:read('train_output'):all():long()

   local sparse_validation_input = f:read('valid_sparse_input'):all():long()
   local dense_validation_input = f:read('valid_dense_input'):all():double()
   local validation_output = f:read('valid_output'):all():long()
   --local word_embeddings = f:read('word_embeddings'):all():double()

   print("Imported all data")

   -- Train.
   --W, b = naiveBayes(sparse_training_input, dense_training_input, training_output, nsparsefeatures, nclasses, 1)
   --print(validateLinearModel(W, b, sparse_validation_input, dense_validation_input, validation_output, nsparsefeatures, ndensefeatures))
   local model = LogisticRegression(sparse_training_input, dense_training_input, training_output, 
   	                  sparse_validation_input, dense_validation_input, validation_output, 
   	                  nsparsefeatures, nclasses, opt.minibatch, opt.eta, opt.epochs, opt.lambda, "nnfig1", opt.hiddenlayers, opt.optimizer, opt.lossfunction, word_embeddings)
   print("Options and accuracy")
   printoptions(opt)
   print(getaccuracy(model, sparse_validation_input, dense_validation_input, validation_output))


   -- Test.
end

main()
