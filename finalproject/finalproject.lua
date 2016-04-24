-- Only requirement allowed
require("hdf5")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'laplace', 'classifier to use')
cmd:option('-b', 128, 'Total number of sequences to split into (for rnn only)')
cmd:option('-alpha', 1, 'laplacian smoothing factor')
cmd:option('-odyssey', false, 'Set to true if running on odyssey')
cmd:option('-sequence_length', 100, 'Length of sequence in batch (for rnn only)')
cmd:option('-embedding_size', 50, 'Size of embeddings')
cmd:option('-optimizer', 'sgd', 'optimizer to use')
cmd:option('-epochs', 10, 'Number of epochs')
cmd:option('-hidden', 50, 'Hidden layer (for nn only)')
cmd:option('-eta', .1, 'Learning rate (for nn and rnn)')
cmd:option('-hacks_wanted', false, 'Enable the hacks')
cmd:option('-rnn_unit1', 'lstm', 'Determine which recurrent unit to use (lstm or gru) for 1st layer - for classifier=rnn only')
cmd:option('-rnn_unit2', 'none', 'Determine which recurrent unit to use (none lstm or gru) for 2nd layer - for classifier=rnn only')
cmd:option('-dropout', .5, 'Dropout probability, only for classifier=rnn, and if rnn_unit2 is not none')
cmd:option('-testfile', '', 'test file')
cmd:option('-cuda', false, 'Set to use cuda')
cmd:option('-minibatch_size', 320, 'Size of minibatches')



function main() 
	-- Parse input params
	opt = cmd:parse(arg)
	_G.path = opt.odyssey and '/n/home09/ankitgupta/CS287/CS287assignments/finalproject/' or ''

	dofile(_G.path..'neural.lua')
	dofile(_G.path..'utils.lua')
	dofile(_G.path..'hmm.lua')
	dofile(_G.path..'memm.lua')

	local f = hdf5.open(opt.datafile, 'r')
	local dwin = f:read('dwin'):all():long()[1]
	local nclasses = f:read('nclasses'):all():long()[1]
	local start_class = f:read('start_idx'):all():long()[1]
	local vocab_size = f:read('vocab_size'):all():long()[1] + 10
	print("Num classes:", nclasses)
	print("Vocab size:", vocab_size)
	print("Start class:", start_class)
	--local start_idx = f:read('start_idx'):all():long()[1]
	--local end_indx = f:read('end_indx'):all():long()[1]

	-- Count based laplace
	local flat_train_input = f:read('train_input'):all():long()
	local flat_train_output = f:read('train_output'):all():long()


	local test_input = f:read('test_input'):all():long()
	local test_output = f:read('test_output'):all():long()

	if opt.cuda then
		require 'cunn'
		cutorch.setDevice(1)
		print("Using cuda")
		flat_train_input = flat_train_input:cuda()
		flat_train_output = flat_train_output:cuda()
		test_input = test_input:cuda()
		test_output = test_output:cuda()	
	end

	print(flat_train_input:size())


	--printoptions(opt)

	--print(flat_valid_output:narrow(1, 1, 20))
	if (opt.classifier == 'rnn') then
		train_input, train_output = reshape_inputs(opt.b, flat_train_input, flat_train_output)
		print(train_input:size())
		print(train_output:size())
		model, crit, embedding = rnn_model(vocab_size, opt.embedding_size, nclasses, opt.rnn_unit1, opt.rnn_unit2, opt.dropout, opt.cuda)
		trainRNN(model,crit,embedding,train_input,train_output,opt.sequence_length, opt.epochs,opt.optimizer,opt.eta,opt.hacks_wanted)
   		testRNN(model, crit, test_input)
   	elseif (opt.classifier == 'hmm') then
   		predictor = hmm_train(flat_train_input:squeeze(), flat_train_output, vocab_size, nclasses, opt.alpha)
   		test_predicted_output = viterbi(test_input, predictor, nclasses, start_class)
   		print(test_predicted_output)
   		acc = prediction_accuracy(test_predicted_output:long(), test_output)
   		print("Accuracy:", acc)
   	elseif (opt.classifier == 'memm') then
   		local model = train_memm(flat_train_input, nil, flat_train_output, 
						vocab_size, 0, nclasses, opt.embedding_size, opt.epochs, opt.minibatch_size, opt.eta, opt.optimizer, opt.hidden)

		predictor = make_predictor_function_memm(model, vocab_size)
   		test_predicted_output = viterbi(test_input, predictor, nclasses, start_class)
   		print(test_predicted_output)
   		acc = prediction_accuracy(test_predicted_output:long(), test_output)
   		print("Accuracy:", acc)
   	else
   		print("Unknown classifier ", opt.classifier)
   	end
end
main()







