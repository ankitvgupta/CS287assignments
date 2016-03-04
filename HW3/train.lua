dofile(_G.path.."utils.lua")
dofile(_G.path.."models.lua")

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
	print("Starting Validation accuracy", getaccuracy(model, validation_input, validation_options, validation_true_out))

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
		print("Epoch "..i.." Validation accuracy:", getaccuracy(model, validation_input, validation_options, validation_true_out))
	end
return model
end
