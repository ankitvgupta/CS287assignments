require 'nn'
require 'rnn'
require 'optim'


function trainMEMM(training_input, 
	               training_output, 
				   model, 
				   crit,
				   num_epochs, 
				   minibatch_size, 
				   eta, 
				   optimizer)

	local ntrainingsamples = training_input:size(1)
	print("Num training samples", ntrainingsamples)

	local parameters, gradParameters = model:getParameters()
	for i = 1, num_epochs do
		print("Beginning epoch", i)

		for j = 2, ntrainingsamples-minibatch_size, minibatch_size do
		    -- zero out our gradients
		    gradParameters:zero()
		    model:zeroGradParameters()

		   	minibatch_inputs = {training_input:narrow(1, j, minibatch_size), training_output:narrow(1, j-1, minibatch_size):reshape(minibatch_size, 1)}
		    minibatch_outputs = training_output:narrow(1, j, minibatch_size)

		    -- Create a closure for optim
		    local feval = function(x)
				if x ~= parameters then
					parameters:copy(x)
				end

				-- reset gradients
				gradParameters:zero()
				preds = model:forward(minibatch_inputs)
				loss = criterion:forward(preds, minibatch_outputs)
				if j == 2 then
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
	    		}
	    		optim.adagrad(feval, parameters, config)
	    	elseif optimizer == "sgd" then
	    		config = {
	    			learningRate = eta, 
	    		}
	    		optim.sgd(feval, parameters, config)
		    else 
		    	print("Invalid optimizer")
		    	assert(false)
		    end

		end
	end

	return model

end

function make_predictor_function_memm(model)

	local predictor = function(c_prev, x_i)
		return torch.exp(model:forward({x_i, torch.LongTensor{c_prev}:reshape(1, 1)})):squeeze()
	end
	return predictor

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

	ntrainingsamples = training_input:size(2)

	start_idx = 1
	if include_prev_classes then
		start_idx = 2
	end

	for i = 1, num_epochs do
		for j = start_idx, ntrainingsamples-seq_length, seq_length do
		    -- zero out our gradients
		    gradParameters:zero()
		    model:zeroGradParameters()
		   	
		    if include_prev_classes then
		    	minibatch_inputs = {training_input:narrow(2, j, seq_length), training_output:narrow(2, j-1, seq_length):reshape(training_output:size(1), seq_length, 1)}
		    else
		   		minibatch_inputs = training_input:narrow(2, j, seq_length)
			end

			local minibatch_outputs = training_output:narrow(2, j, seq_length):t()
			
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

				if j == start_idx then
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
		    	print("Unknown optimization technique")
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
end

function testRNNMEMM(lstm_model, output_model, prev_class_model, test_input, nclasses, start_class, usecuda)
	-- define the memm predictor
	-- prediction model concats lstm output and prev_class rep
	prediction_model = nn.Sequential()
	parallel_table = nn.ParallelTable()
	parallel_table:add(nn.Sequential()) -- takes the lstm output
	parallel_table:add(prev_class_model:add(nn.Squeeze())) -- takes the previous class
	prediction_model:add(parallel_table)
	prediction_model:add(nn.JoinTable(1, 1)) -- concat lstm out and prev class
	prediction_model:add(output_model) -- linear and softmax

	if usecuda then
		prediction_model:cuda()
		print("Converted prediction model into CUDA")
	end

	predictor = function(c_prev, x_i)
		this_input = {x_i, torch.LongTensor{c_prev}:reshape(1, 1, 1)} 
		return torch.exp(prediction_model:forward(this_input))
	end

	-- convert raw input into lstm representation
	print("Running LSTM on test data...")
	lstm_encoded = lstm_model:forward(test_input)

	return viterbi(lstm_encoded:squeeze(), predictor, nclasses, start_class)

end