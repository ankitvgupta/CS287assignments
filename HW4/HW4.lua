-- Only requirement allowed
require("hdf5")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'laplace', 'classifier to use')
cmd:option('-window_size', 5, 'Size of the window to split the full matrix')
cmd:option('-b', 50, 'Total number of sequences to split into (batch)')
cmd:option('-alpha', 1, 'laplacian smoothing factor')
cmd:option('-odyssey', false, 'Set to true if running on odyssey')
cmd:option('-l', 100, 'Length of sequence in batch')
-- Hyperparameters
-- ...


function main() 
   	-- Parse input params
   	opt = cmd:parse(arg)

	_G.path = opt.odyssey and '/n/home09/ankitgupta/CS287/CS287assignments/HW3/' or ''

	dofile(_G.path..'multinomial.lua')
	dofile(_G.path..'utils.lua')


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
   	
   	training_input, training_output = unroll_inputs(flat_train_input, flat_train_output, opt.window_size)
   	valid_input, valid_output = unroll_inputs(flat_valid_input, flat_valid_output, opt.window_size)

   	local reverse_trie = fit(training_input, training_output)
	local log_predictions = getlaplacepredictions(reverse_trie, valid_input, nclasses, opt.alpha)
	local cross_entropy_loss = cross_entropy_loss(valid_output, log_predictions)
	print("Cross-entropy loss", cross_entropy_loss)


   -- Train.

   -- Test.
end

main()
