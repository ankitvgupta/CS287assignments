require("hdf5")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'laplace', 'classifier to use')
cmd:option('-b', 128, 'Total number of sequences to split into (for rnn only)')
cmd:option('-alpha', 1, 'laplacian smoothing factor')
cmd:option('-odyssey', false, 'Set to true if running on odyssey')
cmd:option('-tomodyssey', false, 'Set to true if running on toms odyssey')
cmd:option('-sequence_length', 100, 'Length of sequence in batch (for rnn only)')
cmd:option('-embedding_size', 50, 'Size of embeddings')
cmd:option('-optimizer', 'sgd', 'optimizer to use')
cmd:option('-epochs', 10, 'Number of epochs')
cmd:option('-hidden', 50, 'Hidden layer (for nn and bidirectional rnn only)')
cmd:option('-eta', .1, 'Learning rate (for nn and rnn)')
cmd:option('-hacks_wanted', false, 'Enable the hacks')
cmd:option('-rnn_unit1', 'lstm', 'Determine which recurrent unit to use (lstm or gru) for 1st layer - for classifier=rnn only')
cmd:option('-rnn_unit2', 'none', 'Determine which recurrent unit to use (none lstm or gru) for 2nd layer - for classifier=rnn only')
cmd:option('-dropout', .5, 'Dropout probability, only for classifier=rnn, and if rnn_unit2 is not none')
cmd:option('-testfile', '', 'test file')
cmd:option('-cuda', false, 'Set to use cuda')
cmd:option('-minibatch_size', 320, 'Size of minibatches')
cmd:option('-bidirectional', false, 'Use a bidirectional RNN.')
cmd:option('-bidirectional_layers', 1, 'Number of layers of bidirectional RNN')
cmd:option('-additional_features', false, 'Use additional features.')
cmd:option('-memm_layer', false, 'Use MEMM on top (LSTM with additional features only).')


function main() 
	-- Parse input params
	opt = cmd:parse(arg)
	_G.path = opt.odyssey and '/n/home09/ankitgupta/CS287/CS287assignments/finalproject/' or ''
	_G.path = opt.tomodyssey and '/n/home11/tsilver/CS287/CS287assignments/finalproject/' or ''
	if opt.cuda then
		require 'cunn'
		cutorch.setDevice(1)
		cutorch.setHeapTracking(true)
		print("Using cuda")
	end

	dofile(_G.path..'models.lua')
	dofile(_G.path..'modelrunner.lua')
	dofile(_G.path..'utils.lua')
	printoptions(opt)

	local f = hdf5.open(opt.datafile, 'r')
	local dwin = f:read('dwin'):all():long()[1]
	local nclasses = f:read('nclasses'):all():long()[1]
	local start_class = f:read('start_idx'):all():long()[1]
	local vocab_size = f:read('vocab_size'):all():long()[1] + 10
	print("Num classes:", nclasses)
	print("Vocab size:", vocab_size)

	local flat_train_input = f:read('train_input'):all()
	if opt.additional_features == false then
		flat_train_input = flat_train_input:long()
	end
	local flat_train_output = f:read('train_output'):all():long()

	local num_features = flat_train_input:size(2)
	print("Num features", num_features)

	local test_input = f:read('test_input'):all()
	if opt.additional_features == false then
		test_input = test_input:long()
	end
	local test_output = f:read('test_output'):all():long()
	
	if opt.cuda then
		print("Using cuda")
		flat_train_input = flat_train_input:cuda()
		flat_train_output = flat_train_output:cuda()
		test_input = test_input:cuda()
		test_output = test_output:cuda()	
	end

	local model = nil
	local crit = nil
	local embedding = nil
	local bisequencer_modules = nil

	if (opt.classifier == 'rnn') then

		local desired_test_length = test_input:size(1) - (test_input:size(1) % opt.sequence_length)
		test_input = test_input:narrow(1, 1, desired_test_length)
		test_output = test_output:narrow(1, 1, desired_test_length)
		train_input, train_output = reshape_inputs(opt.b, flat_train_input, flat_train_output)
		test_input, test_output = reshape_inputs(1, test_input, test_output)
		print(test_input:size(), test_output:size())
		print("Data sizes")
		print(train_input:size())
		print(train_output:size())
		if opt.bidirectional then
			if (opt.additional_features and opt.memm_layer) then
				model, lstm_model, output_model, prev_class_model, crit, bisequencer_modules = biLSTMMEMM(num_features, opt.embedding_size, nclasses, opt.rnn_unit1, opt.rnn_unit2, opt.dropout, opt.cuda, opt.hidden, opt.bidirectional_layers)
			elseif opt.additional_features then
				model, crit, bisequencer_modules = bidirectionalRNNmodelExtraFeatures(num_features, opt.embedding_size, nclasses, opt.rnn_unit1, opt.rnn_unit2, opt.dropout, opt.cuda, opt.hidden, opt.bidirectional_layers)
			else
				model, crit, bisequencer_modules = bidirectionalRNNmodel(vocab_size, opt.embedding_size, nclasses, opt.rnn_unit1, opt.rnn_unit2, opt.dropout, opt.cuda, opt.hidden, opt.bidirectional_layers, dwin)
			end
		else
			model, crit, embedding = rnn_model(vocab_size, opt.embedding_size, nclasses, opt.rnn_unit1, opt.rnn_unit2, opt.dropout, opt.cuda)
		end

		model:remember("both")
      	model:training()
		trainRNN(model,crit,embedding,train_input,train_output,opt.sequence_length, opt.epochs,opt.optimizer,opt.eta,opt.hacks_wanted, opt.bidirectional, bisequencer_modules, opt.memm_layer)
   		
   		print("Starting the testing")
   		model:evaluate()

   		if opt.memm_layer then
   			preds = testRNNMEMM(lstm_model, output_model, prev_class_model, test_input, nclasses, start_class, opt.cuda)
   		else
			preds = testRNN(model, crit, test_input, opt.sequence_length, nclasses, opt.bidirectional, bisequencer_modules)
   		end
   		accuracy = torch.sum(torch.eq(preds:double(),test_output:double()))/preds:size(1)
		printoptions(opt)
		print("Accuracy", accuracy)
   	elseif (opt.classifier == 'memm') then
		model, crit = memm_model(num_features, nclasses, opt.embedding_size, opt.hidden)
		trainMEMM(flat_train_input, flat_train_output, model, crit, opt.epochs, opt.minibatch_size, opt.eta, opt.optimizer)

		predictor = make_predictor_function_memm(model, num_features)
   		test_predicted_output = viterbi(test_input, predictor, nclasses, start_class)
   		acc = prediction_accuracy(test_predicted_output:long(), test_output)
   		printoptions(opt)
   		print("Accuracy:", acc)
   	else
   		print("Unknown classifier ", opt.classifier)
   	end
end
main()







