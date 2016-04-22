require 'nn'
require 'rnn'
require 'optim'


function rnn_model(vocab_size, embed_dim, output_dim, rnn_unit1, rnn_unit2, dropout)
	batchLSTM = nn.Sequential()
	local embedding = nn.LookupTableMaskZero(vocab_size, embed_dim)
	batchLSTM:add(embedding) --will return a sequence-length x batch-size x embedDim tensor
	
	-- Transpose the batch and sequence dimensions. This is because we want the splittable to split up elements along the sequence.
	batchLSTM:add(nn.Transpose({1,2}))

	-- 1 indicates the dimension we are splitting along. 3 indicates the number of dimensions in the input (allows for batching)
	batchLSTM:add(nn.SplitTable(1, 3)) --splits into a sequence-length table with batch-size x embedDim entries

	-- Add the first layer rnn unit
	if rnn_unit1 == 'lstm' then
		batchLSTM:add(nn.Sequencer(nn.MaskZero(nn.FastLSTM(embed_dim, embed_dim))))
		print("Unit1: LSTM added")
	elseif rnn_unit1 == 'gru' then 
		batchLSTM:add(nn.Sequencer(nn.MaskZero(nn.GRU(embed_dim, embed_dim))))
		print("Unit1: GRU added")
	else
		print("Invalid unit 1")
		assert(false)
	end

	-- If there is a second layer, add dropout and the layer.
	if rnn_unit2 ~= 'none' then
		batchLSTM:add(nn.Sequencer(nn.MaskZero(nn.Dropout(dropout))))
		print("Dropout added", dropout)
		-- Add second layer 
		if rnn_unit2 == 'lstm' then
			batchLSTM:add(nn.Sequencer(nn.MaskZero(nn.FastLSTM(embed_dim, embed_dim))))
			print("Unit2: LSTM added")
		elseif rnn_unit2 == 'gru' then 
			batchLSTM:add(nn.Sequencer(nn.MaskZero(nn.GRU(embed_dim, embed_dim))))
			print("Unit2: GRU added")
		else
			print("Invalid unit 2")
			assert(false)
		end
	else
		print("No unit 2")
	end

	batchLSTM:add(nn.SelectTable(-1)) -- selects last state of the LSTM

	
	batchLSTM:add(nn.MaskZero(nn.Linear(embed_dim, output_dim)))

	-- Add a criterion
	crit = nn.MultiLabelMarginCriterion()

	return batchLSTM, crit, embedding
end

function nn_model(vocab_size, embedding_dim, window_size, hidden_size, output_dim)
	local model = nn.Sequential()
	model:add(nn.LookupTable(vocab_size, embed_dim))

	filter_width = 4
	model:add(nn.TemporalConvolution(embed_dim, embed_dim, filter_width))
	model:add(nn.TemporalMaxPooling(filter_width))
	model:add(nn.Tanh())
	model:add(nn.TemporalConvolution(embed_dim, embed_dim, filter_width))
	model:add(nn.TemporalMaxPooling(filter_width))
	model:add(nn.Tanh())
	model:add(nn.Linear(embed_dim, output_dim))
	--model:add(nn.LogSoftMax())

	crit = nn.MultiLabelMarginCriterion()

	return model, crit
end

function trainRNN(model,
				criterion,
				embedding,
				training_input,
				training_output,
				minibatch_size, 
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
		for j = 1, training_input:size(1)-minibatch_size, minibatch_size do

		    -- zero out our gradients
		    gradParameters:zero()
		    model:zeroGradParameters()

		    -- Grab the minibatch
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

				-- if gradParameters:norm() > 5 then
				-- 	gradParameters:div(gradParameters:norm()/5)
				-- end


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



-- function nn_model(vocab_size, embedding_dim, window_size, hidden_size, output_dim)
-- 	local model = nn.Sequential()
-- 	local embedding = nn.LookupTable(vocab_size, embedding_dim)
-- 	model:add(embedding)
-- 	model:add(nn.View(-1):setNumInputDims(2))
-- 	model:add(nn.Linear(embedding_dim*window_size, hidden_size))
-- 	model:add(nn.HardTanh())
-- 	model:add(nn.Linear(hidden_size, output_dim))
-- 	model:add(nn.LogSoftMax())
-- 	criterion = nn.ClassNLLCriterion()
-- 	return model, criterion, embedding

-- end

-- function rnn(vocab_size, embed_dim, output_dim, rnn_unit1, rnn_unit2, dropout)
-- 	batchLSTM = nn.Sequential()
-- 	local embedding = nn.LookupTable(vocab_size, embed_dim)
-- 	batchLSTM:add(embedding) --will return a sequence-length x batch-size x embedDim tensor

-- 	-- 1 indicates the dimension we are splitting along. 3 indicates the number of dimensions in the input (allows for batching)
-- 	batchLSTM:add(nn.SplitTable(1, 3)) --splits into a sequence-length table with batch-size x embedDim entries

-- 	-- Add the first layer rnn unit
-- 	if rnn_unit1 == 'lstm' then
-- 		batchLSTM:add(nn.Sequencer(nn.FastLSTM(embed_dim, embed_dim)))
-- 		print("Unit1: LSTM added")
-- 	elseif rnn_unit1 == 'gru' then 
-- 		batchLSTM:add(nn.Sequencer(nn.GRU(embed_dim, embed_dim)))
-- 		print("Unit1: GRU added")
-- 	else
-- 		print("Invalid unit 1")
-- 		assert(false)
-- 	end

-- 	-- If there is a second layer, add dropout and the layer.
-- 	if rnn_unit2 ~= 'none' then
-- 		batchLSTM:add(nn.Sequencer(nn.Dropout(dropout)))
-- 		print("Dropout added", dropout)
-- 		-- Add second layer 
-- 		if rnn_unit2 == 'lstm' then
-- 			batchLSTM:add(nn.Sequencer(nn.FastLSTM(embed_dim, embed_dim)))
-- 			print("Unit2: LSTM added")
-- 		elseif rnn_unit2 == 'gru' then 
-- 			batchLSTM:add(nn.Sequencer(nn.GRU(embed_dim, embed_dim)))
-- 			print("Unit2: GRU added")
-- 		else
-- 			print("Invalid unit 2")
-- 			assert(false)
-- 		end
-- 	else
-- 		print("No unit 2")
-- 	end
	
-- 	-- batchLSTM:add(nn.Sequencer(nn.Dropout(dropout)))
-- 	-- batchLSTM:add(nn.Sequencer(nn.FastLSTM(embed_dim, embed_dim)))
-- 	-- print("Added another dropout and LSTM layer")
	
-- 	batchLSTM:add(nn.Sequencer(nn.Linear(embed_dim, output_dim)))
-- 	batchLSTM:add(nn.Sequencer(nn.LogSoftMax()))

-- 	-- Add a criterion
-- 	crit = nn.SequencerCriterion(nn.ClassNLLCriterion())

-- 	return batchLSTM, crit, embedding
-- end
