require 'nn'
require 'rnn'
require 'optim'

--require 'cudnn'

-- This expects inputs to NOT BE transposed. For example, if you have b sequences of length l, where at each step you are looking 
-- at a window of size dwin, the dimensions of what should be passed into this model are b x l x dwin.
function bidirectionalRNNmodel(vocab_size, embed_dim, output_dim, rnn_unit1, rnn_unit2, dropout, usecuda, hidden, num_layers, dwin)
	batchLSTM = nn.Sequential()

	-- This is needed to deal with SplitTable being stupid about LongTensors
	local copy = nn.Copy('torch.LongTensor', 'torch.DoubleTensor')
	local firstsplit = nn.SplitTable(1,3)
	-- This is needed to deal with LookupTable not having updateGradOutput
	copy.updateGradInput = function() end
	firstsplit.updateGradInput = function() end

	batchLSTM:add(copy)
	batchLSTM:add(firstsplit)
	--batchLSTM:add(nn.Copy('torch.LongTensor', 'torch.DoubleTensor'))
	--batchLSTM:add(nn.SplitTable(1, 3))	
	local embedding = nn.Sequencer(nn.LookupTable(vocab_size, embed_dim))
	batchLSTM:add(embedding) --will return a sequence-length x batch-size x embedDim tensor
	batchLSTM:add(nn.Sequencer(nn.View(-1):setNumInputDims(2)))
	batchLSTM:add(nn.Sequencer(nn.Unsqueeze(2)))
	batchLSTM:add(nn.JoinTable(1, 2))

	batchLSTM:add(nn.SplitTable(1, 3)) --splits into a sequence-length table with batch-size x embedDim entries
	

	local sequencers = {}

	-- local biseq = nn.BiSequencer(nn.FastLSTM(embed_dim, embed_dim), nn.FastLSTM(embed_dim, embed_dim))
	-- batchLSTM:add(biseq)
	-- batchLSTM:add(nn.Sequencer(nn.Dropout(dropout)))
	biseq = nil
	for layer = 1, num_layers do 
		if layer == 1 then 
			biseq = nn.BiSequencer(nn.FastLSTM(dwin*embed_dim, dwin*embed_dim), nn.FastLSTM(dwin*embed_dim, dwin*embed_dim))
		else
			biseq = nn.BiSequencer(nn.FastLSTM(2*dwin*embed_dim, dwin*embed_dim), nn.FastLSTM(2*dwin*embed_dim, dwin*embed_dim))
		end
		sequencers[layer] = biseq
		batchLSTM:add(biseq)
		batchLSTM:add(nn.Sequencer(nn.Dropout(dropout)))
	end
	--batchLSTM:add(nn.BiSequencer(nn.FastLSTM(2*embed_dim, embed_dim), nn.FastLSTM(2*embed_dim, embed_dim)))
	--batchLSTM:add(nn.Sequencer(nn.Dropout(dropout)))
	--batchLSTM:add(nn.BiSequencer(nn.FastLSTM(2*embed_dim, embed_dim), nn.FastLSTM(2*embed_dim, embed_dim)))
	batchLSTM:add(nn.Sequencer(nn.Linear(2*dwin*embed_dim, hidden)))
	batchLSTM:add(nn.Sequencer(nn.ReLU()))
	batchLSTM:add(nn.Sequencer(nn.Dropout(dropout)))
	batchLSTM:add(nn.Sequencer(nn.Linear(hidden, output_dim)))
	batchLSTM:add(nn.Sequencer(nn.LogSoftMax()))

	crit = nn.SequencerCriterion(nn.ClassNLLCriterion())
	if usecuda then
		--cudnn.convert(batchLSTM, cudnn)
		batchLSTM:cuda()
		print("Converted LSTM to CUDA")
		--cudnn.convert(crit, cudnn)
		crit:cuda()
		print("Converted crit to CUDA")
	end
	print(batchLSTM)
	return batchLSTM, crit, sequencers
end


-- This expects inputs to NOT BE transposed. 
-- The input's should be batched though. In particular, the input to this model should be
-- b x n x w, where b is the number of batches, n is the sequence in the batch, and w is the # of features for each seq element.
function bidirectionalRNNmodelExtraFeatures(num_features, embed_dim, output_dim, rnn_unit1, rnn_unit2, dropout, usecuda, hidden, num_layers)
	batchLSTM = nn.Sequential()
	batchLSTM:add(nn.Transpose{1,2})
	batchLSTM:add(nn.SplitTable(1,3))

	batchLSTM:add(nn.Sequencer(nn.Linear(num_features, embed_dim)))


	local sequencers = {}

	-- local biseq = nn.BiSequencer(nn.FastLSTM(embed_dim, embed_dim), nn.FastLSTM(embed_dim, embed_dim))
	-- batchLSTM:add(biseq)
	-- batchLSTM:add(nn.Sequencer(nn.Dropout(dropout)))
	biseq = nil
	for layer = 1, num_layers do 
		if layer == 1 then 
			biseq = nn.BiSequencer(nn.FastLSTM(embed_dim, embed_dim), nn.FastLSTM(embed_dim, embed_dim))
		else
			biseq = nn.BiSequencer(nn.FastLSTM(2*embed_dim, embed_dim), nn.FastLSTM(2*embed_dim, embed_dim))
		end
		sequencers[layer] = biseq
		batchLSTM:add(biseq)
		batchLSTM:add(nn.Sequencer(nn.Dropout(dropout)))
	end
	--batchLSTM:add(nn.BiSequencer(nn.FastLSTM(2*embed_dim, embed_dim), nn.FastLSTM(2*embed_dim, embed_dim)))
	--batchLSTM:add(nn.Sequencer(nn.Dropout(dropout)))
	--batchLSTM:add(nn.BiSequencer(nn.FastLSTM(2*embed_dim, embed_dim), nn.FastLSTM(2*embed_dim, embed_dim)))
	batchLSTM:add(nn.Sequencer(nn.Linear(2*embed_dim, hidden)))
	batchLSTM:add(nn.Sequencer(nn.ReLU()))
	batchLSTM:add(nn.Sequencer(nn.Dropout(dropout)))
	batchLSTM:add(nn.Sequencer(nn.Linear(hidden, output_dim)))
	batchLSTM:add(nn.Sequencer(nn.LogSoftMax()))

	crit = nn.SequencerCriterion(nn.ClassNLLCriterion())
	if usecuda then
		--cudnn.convert(batchLSTM, cudnn)
		batchLSTM:cuda()
		print("Converted LSTM to CUDA")
		--cudnn.convert(crit, cudnn)
		crit:cuda()
		print("Converted crit to CUDA")
	end
	print(batchLSTM)
	return batchLSTM, crit, sequencers
end

-- This expects inputs to NOT BE transposed. 
-- The input's should be batched though. In particular, the input to this model should be
-- b x n x w, where b is the number of batches, n is the sequence in the batch, and w is the # of features for each seq element.
function bidirectionalRNNmodelExtraFeaturesMEMM(num_features, embed_dim, output_dim, rnn_unit1, rnn_unit2, dropout, usecuda, hidden, num_layers)

	parallel_table = nn.ParallelTable()

	prev_class_part = nn.Sequential()
	-- This is needed to deal with SplitTable being stupid about LongTensors
	local copy = nn.Copy('torch.LongTensor', 'torch.DoubleTensor')
	local firstsplit = nn.SplitTable(2,3)
	-- This is needed to deal with LookupTable not having updateGradOutput
	copy.updateGradInput = function() end
	firstsplit.updateGradInput = function() end

	prev_class_part:add(copy)
	prev_class_part:add(firstsplit)
	prev_class_part:add(nn.Sequencer(nn.LookupTable(output_dim, output_dim)))
	prev_class_part:add(nn.Sequencer(nn.View(-1):setNumInputDims(2)))
	prev_class_part:add(nn.Sequencer(nn.Unsqueeze(1)))
	prev_class_part:add(nn.JoinTable(1, 3))

	batchLSTM = nn.Sequential()
	batchLSTM:add(nn.SplitTable(2,3))
	batchLSTM:add(nn.Sequencer(nn.Linear(num_features, embed_dim)))

	local sequencers = {}

	biseq = nil
	for layer = 1, num_layers do 
		if layer == 1 then 
			biseq = nn.BiSequencer(nn.FastLSTM(embed_dim, embed_dim), nn.FastLSTM(embed_dim, embed_dim))
		else
			biseq = nn.BiSequencer(nn.FastLSTM(2*embed_dim, embed_dim), nn.FastLSTM(2*embed_dim, embed_dim))
		end
		sequencers[layer] = biseq
		batchLSTM:add(biseq)
		batchLSTM:add(nn.Sequencer(nn.Dropout(dropout)))
	end
	batchLSTM:add(nn.Sequencer(nn.Linear(2*embed_dim, hidden)))

	-- convert back to tensor so we can concat
	batchLSTM:add(nn.Sequencer(nn.Unsqueeze(1)))
	batchLSTM:add(nn.JoinTable(1, 3))

	parallel_table:add(batchLSTM)
	parallel_table:add(prev_class_part)
	lstmMEMM = nn.Sequential()
	lstmMEMM:add(parallel_table)
	lstmMEMM:add(nn.JoinTable(3, 3))

	--split back before linear and softmax
	lstmMEMM:add(nn.SplitTable(1, 3))

	-- add a linear and a softmax
	output_layer = nn.Sequential()
	output_layer:add(nn.Linear(hidden+output_dim, output_dim))
	output_layer:add(nn:LogSoftMax())
	lstmMEMM:add(nn.Sequencer(output_layer))

	crit = nn.SequencerCriterion(nn.ClassNLLCriterion())
	if usecuda then
		lstmMEMM:cuda()
		print("Converted LSTM to CUDA")
		crit:cuda()
		print("Converted crit to CUDA")
	end
	print(lstmMEMM)
	return lstmMEMM, batchLSTM, output_layer, prev_class_part, crit, sequencers
end

function rnn_model(vocab_size, embed_dim, output_dim, rnn_unit1, rnn_unit2, dropout, usecuda) 
	-- if usecuda then
	-- 	require 'cunn'
	-- end
	batchLSTM = nn.Sequential()
	local embedding = nn.LookupTable(vocab_size, embed_dim)
	batchLSTM:add(embedding) --will return a sequence-length x batch-size x embedDim tensor
	batchLSTM:add(nn.Transpose({1,2}))
	batchLSTM:add(nn.SplitTable(1, 3)) --splits into a sequence-length table with batch-size x embedDim entries

	-- Add the first layer rnn unit
	if rnn_unit1 == 'lstm' then
		batchLSTM:add(nn.Sequencer(nn.FastLSTM(embed_dim, embed_dim)))
		print("Unit1: LSTM added")
	elseif rnn_unit1 == 'gru' then 
		batchLSTM:add(nn.Sequencer(nn.GRU(embed_dim, embed_dim)))
		print("Unit1: GRU added")
	else
		print("Invalid unit 1")
		assert(false)
	end

	-- If there is a second layer, add dropout and the layer.
	if rnn_unit2 ~= 'none' then
		batchLSTM:add(nn.Sequencer(nn.Dropout(dropout)))
		print("Dropout added", dropout)
		-- Add second layer 
		if rnn_unit2 == 'lstm' then
			batchLSTM:add(nn.Sequencer(nn.FastLSTM(embed_dim, embed_dim)))
			print("Unit2: LSTM added")
		elseif rnn_unit2 == 'gru' then 
			batchLSTM:add(nn.Sequencer(nn.GRU(embed_dim, embed_dim)))
			print("Unit2: GRU added")
		else
			print("Invalid unit 2")
			assert(false)
		end
	else
		print("No unit 2")
	end
	batchLSTM:add(nn.Sequencer(nn.Linear(embed_dim, output_dim)))
	batchLSTM:add(nn.Sequencer(nn.LogSoftMax()))

	-- Add a criterion
	crit = nn.SequencerCriterion(nn.ClassNLLCriterion())
	if usecuda then
		--cudnn.convert(batchLSTM, cudnn)
		batchLSTM:cuda()
		print("Converted LSTM to CUDA")
		--cudnn.convert(crit, cudnn)
		crit:cuda()
		print("Converted crit to CUDA")
	end
	return batchLSTM, crit, embedding

end



function trainRNN(model,
				criterion,
				embedding,
				training_input,
				training_output,
				seq_length, 
				num_epochs,
				optimizer,
				eta,
				hacks_wanted,
				bidirectional,
				bisequencer_modules,
				include_prev_classes)
	local parameters, gradParameters = model:getParameters()

	-- initialize the parameters between -.05 and .05
	if hacks_wanted then
		parameters:copy(torch.rand(parameters:size())*.1 - .05)
	end
	--embedding.weight:renorm(2, 1, 5)

	ntrainingsamples = training_input:size(2)

	start_idx = 1
	if include_prev_classes then
		start_idx = 2
	end

	print("Input size", ntrainingsamples)
	-- print("Max train index", torch.max(training_input))
	for i = 1, num_epochs do
		--print("Beginning epoch", i)
		--local valid_numbers = rnn_segment_and_count(valid_kaggle_input:narrow(1, 1, 500), model, space_idx, padding_idx)
      	--local mse = (valid_numbers - valid_kaggle_output:narrow(1, 1, 500)):double():pow(2):mean()
      	--print("MSE", mse)
		--collectgarbage()
		for j = start_idx, ntrainingsamples-seq_length, seq_length do
			--print("Starting at", j)

		    -- zero out our gradients
		    gradParameters:zero()
		    model:zeroGradParameters()
		   	
		    if include_prev_classes then
		    	minibatch_inputs = {training_input:narrow(2, j, seq_length), training_output:narrow(2, j-1, seq_length):reshape(training_output:size(1), seq_length, 1)}
		    else
		   		minibatch_inputs = training_input:narrow(2, j, seq_length)
			end

			local minibatch_outputs = training_output:narrow(2, j, seq_length):t()
			-- print("Shapes")
			-- print(minibatch_inputs:size())
			-- print(minibatch_outputs:size())


			--print("batch", minibatch_inputs)
			--print("model", model)

			--print("Max:", torch.max(minibatch_inputs))
		    -- Create a closure for optim
		    local feval = function(x)
				-- Inspired by this torch demo: https://github.com/andresy/torch-demos/blob/master/train-a-digit-classifier/train-on-mnist.lua
				-- get new parameters
				if x ~= parameters then
					parameters:copy(x)
				end

				-- reset gradients
				gradParameters:zero()

				preds = model:forward(minibatch_inputs)
				-- print("preds", preds)
				--print("outputs", torch.max(minibatch_outputs))
				
				loss = criterion:forward(preds, minibatch_outputs) --+ lambda*torch.norm(parameters,2)^2/2
				--print("Epoch", i, j, loss)
				if j == 1 then
					print("Epoch", i, loss)
				end

				-- backprop
				dLdpreds = criterion:backward(preds, minibatch_outputs) -- gradients of loss wrt preds
				model:backward(minibatch_inputs, dLdpreds)
				if gradParameters:norm() > 5 then
					gradParameters:div(gradParameters:norm()/5)
				end


				return loss,gradParameters
			end

			-- Do the update operation.
	    	if optimizer == "adagrad" then
	    		config =  {
		    		learningRate = eta,
		    		weightDecay = lambda,
		    		learningRateDecay = 5e-7
	    		}
	    		optim.adagrad(feval, parameters, config)
	    	elseif optimizer == "sgd" then
	    		config = {
	    			learningRate = eta, 
	    		}
	    		optim.sgd(feval, parameters, config)
		    else 
		    	assert(false)
		    end
		    -- Make sure the backwards sequence forgets.
		    if bidirectional then
		    	for elem = 1, #bisequencer_modules do
		    		bisequencer_modules[elem].bwdSeq:forget()
		    	end
		    end
		end
	end
end

function testRNN(model, crit, test_input, minibatch_size, nclasses, bidirectional, bisequencer_modules)
	minibatch_size = 5*minibatch_size
	local results = torch.zeros(test_input:size(2), nclasses)
	for j = 1,test_input:size(2)-minibatch_size, minibatch_size do
		local minibatch_input = test_input:narrow(2, j, minibatch_size)
		--local preds = model:forward(test_input:narrow(2, 1, 100000))
		local preds = model:forward(minibatch_input)
		local joined_table = nn.JoinTable(1):forward(preds)
		results:narrow(1, j, minibatch_size):add(joined_table)
	    if bidirectional then
	    	for elem = 1, #bisequencer_modules do
	    		bisequencer_modules[elem].bwdSeq:forget()
	    	end
	    end
	end
	_, i = torch.max(results, 2)
	return i
	--print(joined_table:size())
end

function testRNNMEMM(lstm_model, output_model, prev_class_model, test_input, nclasses, start_class)
	-- define the memm predictor
	-- prediction model concats lstm output and prev_class rep
	prediction_model = nn.Sequential()
	parallel_table = nn.ParallelTable()
	parallel_table:add(nn.Sequential()) -- takes the lstm output
	parallel_table:add(prev_class_model:add(nn.Squeeze())) -- takes the previous class
	prediction_model:add(parallel_table)
	prediction_model:add(nn.JoinTable(1, 1)) -- concat lstm out and prev class
	prediction_model:add(output_model) -- linear and softmax

	predictor = function(c_prev, x_i)
		this_input = {x_i, torch.LongTensor{c_prev}:reshape(1, 1, 1)} 
		return torch.exp(prediction_model:forward(this_input))
	end

	-- convert raw input into lstm representation
	print("Running LSTM on test data...")
	lstm_encoded = lstm_model:forward(test_input)

	print("Starting Viterbi...")
	return viterbi(lstm_encoded:squeeze(), predictor, nclasses, start_class)

end


