-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-testfile', '', 'test file')
cmd:option('-save_losses', '', 'file to save loss per epoch to (leave blank if not wanted)')
cmd:option('-classifier', 'nn', 'classifier to use (mle, nn)')
cmd:option('-alpha', 1, 'laplacian smoothing factor')
cmd:option('-eta', .1, 'Learning rate (.1 for adagrad, 500 for sgd, 10 for nn sgd)')
cmd:option('-lambda', 0, 'regularization penalty (not implemented)')
cmd:option('-minibatch', 2000, 'Minibatch size (500 for nn, 2000 for lr)')
cmd:option('-epochs', 20, 'Number of epochs of SGD')
cmd:option('-optimizer', 'sgd', 'Name of optimizer to use (adagrad or sgd)')
cmd:option('-hiddenlayers', 10, 'Number of hidden layers (if using neural net)')
cmd:option('-embedding_size', 50, 'Size of word embedding')
cmd:option('-odyssey', false, 'Set to true if running on odyssey')

-- Hyperparameters
-- ...


function main() 
   -- Parse input params
   opt = cmd:parse(arg)

   print("Datafile:", opt.datafile, "Classifier:", opt.classifier, "Alpha:", opt.alpha, "Eta:", opt.eta, "Lambda:", opt.lambda, "Minibatch size:", opt.minibatch, "Num Epochs:", opt.epochs, "Optimizer:", opt.optimizer, "Hidden Layers:", opt.hiddenlayers, "Embedding size:", opt.embedding_size)

   _G.path = opt.odyssey and '/n/home09/ankitgupta/CS287/CS287assignments/HW3/' or ''

   dofile(_G.path..'train.lua')

   local f = hdf5.open(opt.datafile, 'r')

   local nclasses = f:read('numClasses'):all():long()[1]
   local nfeatures = f:read('numFeatures'):all():long()[1]
   local d_win = f:read('d_win'):all():long()[1]

   print("nclasses:", nclasses, "nfeatures:", nfeatures, "d_win:", d_win)

   local training_input = f:read('train_input'):all():long()   
   local training_output = f:read('train_output'):all():long()

   local valid_input = f:read('valid_context'):all():long()
   local valid_options = f:read('valid_options'):all():long()
   local valid_true_outs = f:read('valid_true_outs'):all():long()

   local test_context = f:read('test_context'):all():long()
   local test_options = f:read('test_options'):all():long()

   -- Train neural network.
   if opt.classifier == "nn" then
   		local model, criterion = neuralNetwork(nfeatures, opt.hiddenlayers, nclasses, opt.embedding_size, d_win)
   		model = trainModel(model, criterion, training_input, training_output, valid_input, valid_options, valid_true_outs, opt.minibatch, opt.epochs, opt.optimizer, opt.save_losses)
   end

   print("Accuracy:")
   print(getaccuracy(model, valid_input, valid_options, valid_true_outs))



end

main()
