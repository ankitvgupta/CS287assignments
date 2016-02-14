-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

-- For local use, use these.
dofile('nb.lua')
dofile('logisticregression2.lua')

-- For Odyssey, uncomment these
--dofile('/n/home09/ankitgupta/CS287/CS287assignments/HW2/nb.lua')
--dofile('/n/home09/ankitgupta/CS287/CS287assignments/HW2/logisticregression.lua')
--dofile('/n/home09/ankitgupta/CS287/CS287assignments/HW2/logisticregression2.lua')


cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'lr', 'classifier to use')
cmd:option('-alpha', 1, 'laplacian smoothing factor for NB')
cmd:option('-eta', .1, 'Learning rate')
cmd:option('-lambda', 5, 'regularization penalty')
cmd:option('-minibatch', 1000, 'Minibatch size')
cmd:option('-epochs', 10, 'Number of epochs of SGD')

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

   print("Imported all data")

   -- Train.
   --W, b = naiveBayes(sparse_training_input, dense_training_input, training_output, nsparsefeatures, nclasses, 1)
   --print(validateLinearModel(W, b, sparse_validation_input, dense_validation_input, validation_output, nsparsefeatures, ndensefeatures))
   local model = LogisticRegression(sparse_training_input, dense_training_input, training_output, 
   	                  sparse_validation_input, dense_validation_input, validation_output, 
   	                  nsparsefeatures, nclasses, opt.minibatch, opt.eta, opt.epochs, opt.lambda)
   print("Options and accuracy")
   printoptions(opt)
   print(getaccuracy(model, validation_sparse_input, validation_dense_input, validation_output))


   -- Test.
end

main()
