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
					valid_blanks_input, 
					valid_blanks_options,
					valid_blanks_output,
					minibatch_size, 
					num_epochs,
					optimizer, 
					save_losses,
					D_sparse_in,
					D_hidden,
					D_output,
					embedding_size,
					window_size,
					alpha, 
					eta, 
					sample_indices, 
					K, 
					p_ml_tensor,
					valid_input,
					valid_output)

	-- For loss plot.
	file = nil
	if save_losses ~= '' then
		file = io.open(save_losses, 'w')
   		file:write("Epoch,Loss\n")
   	end

   	

	local model, embedding, lookup, bias = NCE(D_sparse_in, D_hidden, D_output, embedding_size, window_size)
	local modelparams, modelgradparams = model:getParameters()

	local lookupparams, lookupgrads = lookup:getParameters()
	local biasparams, biasgradparams = bias:getParameters()
	print(training_input:size())

	local k_index = 1
	for i = 1, num_epochs do
		-- renormalize embedding weights for regularization
		embedding.weight:renorm(embedding.weight, 2, 1, 1)

		print("Epoch", i, "L1 norm of model params:", torch.abs(modelparams):sum(), "LookupParams:", torch.abs(lookupparams):sum(), "Biasparams:", torch.abs(biasparams):sum())
		print("Accuracy, CrossEntropy, Perplexity:", getNCEStats(model, lookup, bias, valid_blanks_input, valid_blanks_options, valid_blanks_output, p_ml_tensor))
		print(NCE_predictions2(model, lookup, bias, valid_input, valid_output, D_hidden, D_output))
		for j = 1, training_input:size(1)-minibatch_size, minibatch_size do

		    -- get the minibatch
		    minibatch_inputs = training_input:narrow(1, j, minibatch_size)
		    minibatch_outputs = training_output:narrow(1, j, minibatch_size)
		    sample_batch = sample_indices:narrow(1, k_index, minibatch_size*K)
		    k_index = (k_index + minibatch_size*K) % (10000000 - minibatch_size*K)

		    forwardandBackwardPass3(model, modelparams, modelgradparams,lookup, lookupparams, lookupgrads, minibatch_inputs, minibatch_outputs, sample_batch, p_ml_tensor, eta, bias, biasparams, biasgradparams, K)
		end
	end

	return model, lookup, bias, embedding
end







