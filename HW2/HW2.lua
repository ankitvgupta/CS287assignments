-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

dofile('nb.lua')

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')

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
   local dense_training_input = f:read('train_dense_input'):all():long()
   local training_output = f:read('train_output'):all():long()

   local sparse_validation_input = f:read('valid_sparse_input'):all():long()
   local dense_validation_input = f:read('valid_dense_input'):all():long()
   local validation_output = f:read('valid_output'):all():long()

   --local W = torch.DoubleTensor(nclasses, nfeatures)
   --local b = torch.DoubleTensor(nclasses)

   -- Train.
   W, b = naiveBayes(sparse_training_input, dense_training_input, training_output, nsparsefeatures, nclasses, 1)
   print(validateLinearModel(W, b, sparse_validation_input, dense_validation_input, validation_output, nsparsefeatures, ndensefeatures))

   -- Test.
end

main()
