require 'nn'
require 'rnn'
require 'optim'

--require 'cudnn'

function bidirectionalRNNmodel(vocab_size, embed_dim, output_dim, rnn_unit1, rnn_unit2, dropout, usecuda, hidden)
	batchLSTM = nn.Sequential()
	local embedding = nn.LookupTable(vocab_size, embed_dim)
	batchLSTM:add(embedding) --will return a sequence-length x batch-size x embedDim tensor
	batchLSTM:add(nn.Transpose({1,2}))
	batchLSTM:add(nn.SplitTable(1, 3)) --splits into a sequence-length table with batch-size x embedDim entries
	
	local biseq = nn.BiSequencer(nn.FastLSTM(embed_dim, embed_dim), nn.FastLSTM(embed_dim, embed_dim))
	batchLSTM:add(biseq)
	batchLSTM:add(nn.Sequencer(nn.Linear(2*embed_dim, hidden)))
	batchLSTM:add(nn.Sequencer(nn.Tanh()))
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
	return batchLSTM, crit, biseq
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
				biseqencer_module)
	local parameters, gradParameters = model:getParameters()

	-- initialize the parameters between -.05 and .05
	if hacks_wanted then
		parameters:copy(torch.rand(parameters:size())*.1 - .05)
	end
	--embedding.weight:renorm(2, 1, 5)

	print("Input size", training_input:size(2))
	print("Max train index", torch.max(training_input))
	for i = 1, num_epochs do
		--print("Beginning epoch", i)
		--local valid_numbers = rnn_segment_and_count(valid_kaggle_input:narrow(1, 1, 500), model, space_idx, padding_idx)
      	--local mse = (valid_numbers - valid_kaggle_output:narrow(1, 1, 500)):double():pow(2):mean()
      	--print("MSE", mse)
		for j = 1, training_input:size(2)-seq_length, seq_length do
			--print("Starting at", j)

		    -- zero out our gradients
		    gradParameters:zero()
		    model:zeroGradParameters()

		   	local minibatch_inputs = training_input:narrow(2, j, seq_length)
			local minibatch_outputs = training_output:narrow(2, j, seq_length):t()
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
				--print("preds", preds)
				--print("outputs", torch.max(minibatch_outputs))
				
				loss = criterion:forward(preds, minibatch_outputs) --+ lambda*torch.norm(parameters,2)^2/2
				--print("Loss", loss)
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
		    	biseqencer_module.bwdSeq:forget()
		    end
		end
	end
end

function testRNN(model, crit, test_input, minibatch_size, nclasses, bidirectional, biseqencer_module)
	minibatch_size = 5*minibatch_size
	local results = torch.zeros(test_input:size(2), nclasses)
	for j = 1,test_input:size(2)-minibatch_size, minibatch_size do
		local minibatch_input = test_input:narrow(2, j, minibatch_size)
		--local preds = model:forward(test_input:narrow(2, 1, 100000))
		local preds = model:forward(minibatch_input)
		local joined_table = nn.JoinTable(1):forward(preds)
		results:narrow(1, j, minibatch_size):add(joined_table)
		if bidirectional then
		    biseqencer_module.bwdSeq:forget()
		end
	end
	_, i = torch.max(results, 2)
	return i
	--print(joined_table:size())
end
