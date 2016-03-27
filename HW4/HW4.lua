-- Only requirement allowed
require("hdf5")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'laplace', 'classifier to use')
cmd:option('-window_size', 5, 'Window size (does not apply to rnn)')
cmd:option('-b', 50, 'Total number of sequences to split into (minibatch_size for nn)')
cmd:option('-alpha', 1, 'laplacian smoothing factor')
cmd:option('-odyssey', false, 'Set to true if running on odyssey')
cmd:option('-sequence_length', 100, 'Length of sequence in batch (for rnn only)')
cmd:option('-embedding_size', 50, 'Size of embeddings')
cmd:option('-optimizer', 'sgd', 'optimizer to use')
cmd:option('-epochs', 10, 'Number of epochs')
cmd:option('-hidden', 50, 'Hidden layer')
-- Hyperparameters
-- ...


function main() 
   -- Parse input params
   opt = cmd:parse(arg)

	_G.path = opt.odyssey and '/n/home09/ankitgupta/CS287/CS287assignments/HW3/' or ''

	dofile(_G.path..'multinomial.lua')
	dofile(_G.path..'utils.lua')
   dofile(_G.path..'neural.lua')



	local f = hdf5.open(opt.datafile, 'r')
	nclasses = f:read('nclasses'):all():long()[1]
	nfeatures = f:read('nfeatures'):all():long()[1]
	space_idx = f:read('spaceIdx'):all():long()[1]

	-- Count based laplace
	flat_train_input = f:read('train_input'):all():long()
	flat_train_output = f:read('train_output'):all():long()
	flat_valid_input = f:read('valid_input'):all():long()
	flat_valid_output = f:read('valid_output'):all():long()	
   print(flat_train_input:size())
   print(flat_train_output:size())
   print(flat_valid_output:size())
	
   if opt.classifier == 'laplace' then
   	local training_input, training_output = unroll_inputs(flat_train_input, flat_train_output, opt.window_size)
   	local valid_input, valid_output = unroll_inputs(flat_valid_input, flat_valid_output, opt.window_size)

   	local reverse_trie = fit(training_input, training_output)
   	local predictions = laplace_greedily_segment(flat_valid_input, reverse_trie, opt.alpha, opt.window_size, space_idx)
   	local accuracy = prediction_accuracy(predictions, flat_valid_output)
   	local precision = prediction_precision(predictions, flat_valid_output)
   	print("Accuracy:", precision)
   elseif opt.classifier == 'neural' then
      local model, crit = nn_model(nfeatures, opt.embedding_size, opt.window_size, opt.hidden, 2)
      local training_input, training_output = unroll_inputs(flat_train_input, flat_train_output, opt.window_size)
      trainNN(model, crit, training_input, training_output, opt.minibatch_size, opt.epochs, opt.optimizer, opt.b)
   elseif opt.classifier == 'rnn' then
      print("RNN")
      local model, crit = rnn(nfeatures, opt.embedding_size, 2)
      local training_input, training_output = create_nn_inputs(flat_train_input, flat_train_output, opt.b)
      trainRNN(model, crit, training_input, training_output, opt.sequence_length,  opt.epochs, opt.optimizer)
   end



   -- Train.

   -- Test.
end

main()
