dofile(_G.path.."utils.lua")
dofile(_G.path.."models.lua")
dofile(_G.path.."test.lua")

function trainModel(model, 
					criterion,
					training_input, 
					training_output,
					validation_input, 
					validation_options,
					validation_true_out,
					minibatch_size, 
					num_epochs,
					optimizer, 
					save_losses)

	-- For loss plot.
	file = nil
	if save_losses ~= '' then
		file = io.open(save_losses, 'w')
   		file:write("Epoch,Loss\n")
   	end

   	local parameters, gradParameters = model:getParameters()

   	print("Got params and grads")
	print("Starting Validation accuracy", getaccuracy2(model, validation_input, validation_options, validation_true_out))

	for i = 1, num_epochs do
		print("L1 norm of params:", torch.abs(parameters):sum())
		for j = 1, training_input:size(1)-minibatch_size, minibatch_size do

		    -- zero out our gradients
		    gradParameters:zero()
		    model:zeroGradParameters()

		    -- get the minibatch
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
				--print(loss)

				-- backprop
				dLdpreds = criterion:backward(preds, minibatch_outputs) -- gradients of loss wrt preds
				model:backward(minibatch_inputs, dLdpreds)

				if j == 1 then
					if save_losses ~= '' then
						file:write(i, ',', loss, '\n')
					else
						print("Loss: ", loss)
					end
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
		--print("Epoch "..i.." Validation accuracy:", getaccuracy(model, validation_input, validation_options, validation_true_out))
		print("Epoch "..i.." Validation accuracy:", getaccuracy2(model, validation_input, validation_options, validation_true_out))
	end
return model
end







function trainNCEModel(
					training_input, 
					training_output,
					validation_input, 
					validation_options,
					validation_true_out,
					minibatch_size, 
					num_epochs,
					optimizer, 
					save_losses,
					D_sparse_in,
					D_hidden,
					D_output,
					embedding_size,
					window_size,
					alpha, eta, sample_indices, K)

	-- For loss plot.
	file = nil
	if save_losses ~= '' then
		file = io.open(save_losses, 'w')
   		file:write("Epoch,Loss\n")
   	end

   	local reverse_trie = fit(training_input, training_output)
   	local distribution = normalize_table(get_word_counts_for_context(reverse_trie, torch.LongTensor{}, D_output, alpha))
   	local p_ml_tensor = table_to_tensor(distribution, D_output)

	local model, lookup, bias = NCE(D_sparse_in, D_hidden, D_output, embedding_size, window_size)
	local modelparams, modelgradparams = model:getParameters()
	--input_batch = torch.LongTensor{{7, 5, 2},{1, 3, 4}}
	--output_batch = torch.LongTensor{1, 2}
	--local sample_indices = torch.LongTensor{1, 72, 21, 341, 17, 8, 15}
	--sample_probs = torch.Tensor{.1, .1, .1, .1, .1, .1, .1, .1, .1, .1}
	local lookupparams, lookupgrads = lookup:getParameters()
	local biasparams, biasgradparams = bias:getParameters()
	print(training_input:size())

   	--local parameters, gradParameters = model:getParameters()

   	--print("Got params and grads")
	--print("Starting Validation accuracy", getaccuracy2(model, validation_input, validation_options, validation_true_out))

	for i = 1, num_epochs do
		print("Epoch", i, "L1 norm of params:", torch.abs(modelparams):sum())
		for j = 1, training_input:size(1)-minibatch_size, minibatch_size do
			--print(j)
		--for j = 1, training_input:size(1), 1 do
			--if (j-1)%1000 == 0 then print(j) end
		    -- zero out our gradients
		    --gradParameters:zero()
		    --model:zeroGradParameters()

		    -- get the minibatch
		    -- minibatch_inputs = training_input:narrow(1, j, minibatch_size)
		    -- minibatch_outputs = training_output:narrow(1, j, minibatch_size)
		    -- sample_batch = training_output:narrow(1, j, K)
		    minibatch_inputs = training_input:narrow(1, j, 1)
		    minibatch_outputs = training_output:narrow(1, j, 1)
		    sample_batch = sample_indices:narrow(1, j*K % (1000000 - K), K)
		    forwardandBackwardPass3(model, modelparams, modelgradparams,lookup, lookupparams, lookupgrads, minibatch_inputs, minibatch_outputs, sample_batch, p_ml_tensor, eta, bias, biasparams, biasgradparams)


		 --    -- Create a closure for optim
		 --    local feval = function(x)
			-- 	-- Inspired by this torch demo: https://github.com/andresy/torch-demos/blob/master/train-a-digit-classifier/train-on-mnist.lua
			-- 	-- get new parameters
			-- 	if x ~= parameters then
			-- 		parameters:copy(x)
			-- 	end
			-- 	-- reset gradients
			-- 	gradParameters:zero()

			-- 	preds = model:forward(minibatch_inputs)
			-- 	loss = criterion:forward(preds, minibatch_outputs) --+ lambda*torch.norm(parameters,2)^2/2
			-- 	--print(loss)

			-- 	-- backprop
			-- 	dLdpreds = criterion:backward(preds, minibatch_outputs) -- gradients of loss wrt preds
			-- 	model:backward(minibatch_inputs, dLdpreds)

			-- 	if j == 1 then
			-- 		if save_losses ~= '' then
			-- 			file:write(i, ',', loss, '\n')
			-- 		else
			-- 			print("Loss: ", loss)
			-- 		end
			-- 	end

			-- 	return loss,gradParameters
			-- end
			
			-- Do the update operation.
	    	-- if optimizer == "adagrad" then
	    	-- 	config =  {
	    	-- 	learningRate = eta,
	    	-- 	weightDecay = lambda,
	    	-- 	learningRateDecay = 5e-7
	    	-- }
	    	-- optim.adagrad(feval, parameters, config)
	    	-- elseif optimizer == "sgd" then
	    	-- 	config = {
	    	-- 	learningRate = eta, 
	    	-- }

	    	-- optim.sgd(feval, parameters, config)
		    -- else 
		    -- 	assert(false)
		    -- end
		    

		end
		--print("Epoch "..i.." Validation accuracy:", getaccuracy(model, validation_input, validation_options, validation_true_out))
		--print("Epoch "..i.." Validation accuracy:", getaccuracy2(model, validation_input, validation_options, validation_true_out))
	end
	print(lookup.weight)
	return model, lookup, bias
end







