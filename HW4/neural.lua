require 'nn'
require 'rnn'
require 'optim'

function nn_model(vocab_size, embedding_dim, window_size, hidden_size, output_dim)
	local model = nn.Sequential()
	local embedding = nn.LookupTable(vocab_size, embedding_dim)
	model:add(embedding)
	model:add(nn.View(-1):setNumInputDims(2))
	model:add(nn.Linear(embedding_dim*window_size, hidden_size))
	model:add(nn.HardTanh())
	model:add(nn.Linear(hidden_size, output_dim))
	model:add(nn.LogSoftMax())
	criterion = nn.ClassNLLCriterion()
	return model, criterion, embedding

end
function rnn(vocab_size, embed_dim, output_dim, rnn_unit1, rnn_unit2, dropout)
	batchLSTM = nn.Sequential()
	local embedding = nn.LookupTable(vocab_size, embed_dim)
	batchLSTM:add(embedding) --will return a sequence-length x batch-size x embedDim tensor

	-- 1 indicates the dimension we are splitting along. 3 indicates the number of dimensions in the input (allows for batching)
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
	
	-- batchLSTM:add(nn.Sequencer(nn.Dropout(dropout)))
	-- batchLSTM:add(nn.Sequencer(nn.FastLSTM(embed_dim, embed_dim)))
	-- print("Added another dropout and LSTM layer")
	
	batchLSTM:add(nn.Sequencer(nn.Linear(embed_dim, output_dim)))
	batchLSTM:add(nn.Sequencer(nn.LogSoftMax()))

	-- Add a criterion
	crit = nn.SequencerCriterion(nn.ClassNLLCriterion())

	return batchLSTM, crit, embedding
end

function trainNN(model, crit, training_input, training_output, minibatch_size, num_epochs, optimizer, minibatch_size, eta)
	local parameters, gradParameters = model:getParameters()

	for i = 1, num_epochs do
		print("Beginning epoch", i)

		for j = 1, training_input:size(1)-minibatch_size, minibatch_size do

		    -- zero out our gradients
		    gradParameters:zero()
		    model:zeroGradParameters()

		   	minibatch_inputs = training_input:narrow(1, j, minibatch_size)
		    minibatch_outputs = training_output:narrow(1, j, minibatch_size)

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
				loss = criterion:forward(preds, minibatch_outputs) --+ lambda*torch.norm(parameters,2)^2/2
				if j == 1 then
					print("    ", loss)
				end

				-- backprop
				dLdpreds = criterion:backward(preds, minibatch_outputs) -- gradients of loss wrt preds
				model:backward(minibatch_inputs, dLdpreds)


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

		end
	end
end


function trainRNN(model,
				criterion,
				embedding,
				training_input,
				training_output,
				valid_kaggle_input,
				valid_kaggle_output,
				space_idx,
				padding_idx,
				l, 
				num_epochs,
				optimizer,
				eta,
				hacks_wanted)
	local parameters, gradParameters = model:getParameters()

	-- initialize the parameters between -.05 and .05
	if hacks_wanted then
		parameters:copy(torch.rand(parameters:size())*.1 - .05)
	end
	--embedding.weight:renorm(2, 1, 5)


	for i = 1, num_epochs do
		--print("Beginning epoch", i)
		--local valid_numbers = rnn_segment_and_count(valid_kaggle_input:narrow(1, 1, 500), model, space_idx, padding_idx)
      	--local mse = (valid_numbers - valid_kaggle_output:narrow(1, 1, 500)):double():pow(2):mean()
      	--print("MSE", mse)
		for j = 1, training_input:size(2)-l, l do

		    -- zero out our gradients
		    gradParameters:zero()
		    model:zeroGradParameters()

		   	minibatch_inputs = training_input:narrow(2, j, l):t()
		    minibatch_outputs = training_output:narrow(2, j, l):t()

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
				loss = criterion:forward(preds, minibatch_outputs) --+ lambda*torch.norm(parameters,2)^2/2
				if j == 1 then
					print("    ", loss)
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

		end
	end
end

function nn_greedily_segment(flat_valid_input, model, window_size, space_idx)

	print("Starting predictions")
	local valid_input_count = flat_valid_input:size(1)
	local valid_output_predictions = torch.ones(valid_input_count):long()
	local next_window = torch.Tensor(window_size):copy(flat_valid_input:narrow(1, 1, window_size))
	local next_word_idx = window_size+1

	while next_word_idx < valid_input_count do 

		if next_word_idx % 1000 == 0 then
			print("Word", next_word_idx)
		end

		local predictions = model:forward(next_window)
		
		-- shift the window
		for i=1, window_size-1 do
			next_window[i] = next_window[i+1]
		end

		-- predicting a space
		if (predictions[2] > torch.log(0.5)) then
			valid_output_predictions[next_word_idx-window_size] = 2
			next_window[window_size] = space_idx
		-- predicting a non-space, so grab the next valid input
		else
			next_window[window_size] = flat_valid_input[next_word_idx]
			next_word_idx = next_word_idx + 1
		end

	end

	return valid_output_predictions

end

function nn_viterbi_segment(flat_valid_input, model, window_size, space_idx)

	print("Starting viterbi")
	local valid_input_count = flat_valid_input:size(1)
	local backpointers = torch.ones(valid_input_count, 2):long()

	-- This is the probabilities if the last one was a space.
	local post_space_probs = model:forward(torch.Tensor{space_idx})

	-- BIGRAMS ONLY
	local next_window = torch.Tensor(1)
	next_window[1] = flat_valid_input[1]

	local pi = torch.ones(valid_input_count, 2):mul(-1e+31)
	pi[1] = model:forward(next_window)
	--pi[1][1] = 0
	--pi[1][2] = 0

	for i = 2, valid_input_count do

		next_window[1] = flat_valid_input[i]
		local yci11 = model:forward(next_window)
			
		-- Last one was not a space, and next is not a space.
		local score1 = pi[i-1][1] + yci11[1]

		-- Last one was not a space, and next is a space.
		local score3 = pi[i-1][1] + yci11[2]

		-- Last one was a space, and next is not a space.
		local score2 = pi[i-1][2] + yci11[1]

		-- Last one was a space, and next one is a space.
		local score4 = pi[i-1][2] + yci11[2]

		-- The argmax thing.
		if score1 > pi[i][1] then
			pi[i][1] = score1
			backpointers[i][1] = 1
		end
		if score2 > pi[i][1] then
			pi[i][1] = score2
			backpointers[i][1] = 2
		end
		if score3 > pi[i][2] then
			pi[i][2] = score3
			backpointers[i][2] = 1
		end
		if score4 > pi[i][2] then
			pi[i][2] = score4
			backpointers[i][2] = 2
		end

	end

	local valid_output_predictions = torch.ones(valid_input_count):long()

	local last_class = 1
	if pi[valid_input_count][2] > pi[valid_input_count][1] then
		last_class = 2
	end

	valid_output_predictions[valid_input_count] = last_class

	for i=valid_input_count-1, 1, -1 do
		valid_output_predictions[i] = backpointers[i+1][valid_output_predictions[i+1]]
	end

	return valid_output_predictions

end

function nn_log_probs(flat_valid_input, model, window_size)
	local valid_input_count = flat_valid_input:size(1)
	local log_probs = torch.Tensor(valid_input_count-1, 2)

	for i=1, valid_input_count-window_size-1 do
		next_window = flat_valid_input:narrow(1, i, window_size)
		log_probs[i] = model:forward(next_window)
	end

	return log_probs

end

function rnn_greedily_segment(flat_valid_input, model, space_idx, padding_idx)

	--print("Starting predictions")
	model:evaluate()
	local valid_input_count = flat_valid_input:size(1)
	local valid_output_predictions = torch.ones(valid_input_count):long()
	local next_word_idx = 1
	local char = flat_valid_input:narrow(1, next_word_idx, 1)

	while next_word_idx < valid_input_count and char[1] ~= padding_idx do 

		if next_word_idx % 1000 == 0 then
			print("Word", next_word_idx)
		end

		local predictions = model:forward(char)[1]
		--print(predictions)

		-- predicting a space
		if (predictions[2] > torch.log(0.5)) then
			--print('Space Found')
			valid_output_predictions[next_word_idx] = 2
			char = torch.LongTensor{space_idx}
		-- predicting a non-space, so grab the next valid input
		else
			next_word_idx = next_word_idx + 1
			char = flat_valid_input:narrow(1, next_word_idx, 1)
		end

	end

	return valid_output_predictions

end

function rnn_segment_and_count(test_input, model, space_idx, padding_idx) 
	local counts = torch.LongTensor(test_input:size(1))
	for i = 1, test_input:size(1) do
		local predictions = rnn_greedily_segment(test_input[i],model, space_idx, padding_idx)
		-- Since predictions is all 1s and 2s, where 2s are the spaces, just subtract 1 and sum it to get
		-- the number of spaces
		counts[i] = (predictions - 1):sum()
	end
	return counts
end

function example()
	r, crit = rnn(100, 5, 2)

	-- batchsize = 5. SequenceLength = 3. 
	inputs = (torch.abs(torch.randn(5,3)*10) + 1):long()
	-- batchSize = 5. These outputs will be either 1 or 2.
	-- There are 3 of these, since ther eis one for every element in sequence.
	batch_output = torch.round(torch.rand(5)) + 1
	outputs = {batch_output, batch_output, batch_output}
	-- Make some predictions
	res = r:forward(inputs:t())
	print(res)
	loss = crit:forward(res, outputs)
	print(loss)
	dLdPreds = crit:backward(res, outputs)
	print(dLdPreds)
	r:backward(inputs:t(), dLdPreds)
end
--example()

