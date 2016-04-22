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
cmd:option('-hidden', 50, 'Hidden layer (for nn only)')
cmd:option('-eta', 1, 'Learning rate (for nn and rnn)')
cmd:option('-hacks_wanted', false, 'Enable the hacks')
cmd:option('-rnn_unit1', 'lstm', 'Determine which recurrent unit to use (lstm or gru) for 1st layer - for classifier=rnn only')
cmd:option('-rnn_unit2', 'none', 'Determine which recurrent unit to use (none lstm or gru) for 2nd layer - for classifier=rnn only')
cmd:option('-dropout', .5, 'Dropout probability, only for classifier=rnn, and if rnn_unit2 is not none')
cmd:option('-testfile', '', 'test file')

function main() 
	-- Parse input params
	opt = cmd:parse(arg)
	_G.path = opt.odyssey and '/n/home09/ankitgupta/CS287/CS287assignments/finalproject/' or ''

	dofile(_G.path..'neural.lua')

	local f = hdf5.open(opt.datafile, 'r')
	local ngrams = f:read('ngrams'):all():long()[1]
	local nclasses = f:read('nclasses'):all():long()[1]
	local vocab_size = f:read('vocab_size'):all():long()[1]

	-- Count based laplace
	local train_input = f:read('train_input'):all():long()
	local train_output = f:read('train_output'):all():long()
	local test_input = f:read('test_input'):all():long()
	local test_output = f:read('test_output'):all():long()
	
	printoptions(opt)

	--print(flat_valid_output:narrow(1, 1, 20))
	if opt.classifier == 'rnn' then
		model, crit = rnn_model(vocab_size, opt.embedding_size, nclasses, opt.rnn_unit1, opt.rnn_unit2, opt.dropout)

   end
end
main()







