-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

dofile('nb.lua')
dofile('logisticregression.lua')

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')
cmd:option('-alpha', 1, 'laplacian smoothing factor for NB')
cmd:option('-eta', 1.0, 'Learning rate')
cmd:option('-lambda', 1, 'regularization penalty')
cmd:option('-minibatch', 500, 'Minibatch size')
cmd:option('-epochs', 50, 'Number of epochs of SGD')
cmd:option('-min_sentence_length', 0, 'Minimum length of sentence to be included in training set')
cmd:option('-test_file', '', 'File to put results from test set. Leave nil if not wanted')
cmd:option('-generate_validation_set', 0, "Set to 1 if validation set needs to be self generated")

-- Hyperparameters
-- ...


function main() 
   -- Parse input params
   opt = cmd:parse(arg)
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

   print("Imported all data")

   --local W = torch.DoubleTensor(nclasses, nfeatures)
   --local b = torch.DoubleTensor(nclasses)

   -- Train.
   --W, b = naiveBayes(sparse_training_input, dense_training_input, training_output, nsparsefeatures, nclasses, 1)
   --print(validateLinearModel(W, b, sparse_validation_input, dense_validation_input, validation_output, nsparsefeatures, ndensefeatures))
   LogisticRegression(sparse_training_input, dense_training_input, training_output, sparse_validation_input, dense_validation_input, validation_output, nsparsefeatures, nclasses, 64, .1, 1)

   -- Test.
end

main()
