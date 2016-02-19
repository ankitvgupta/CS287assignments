require('nn')
--require('cudnn')
--dofile("utils.lua")
--dofile("models.lua")
-- -- For odyssey
dofile("/n/home09/ankitgupta/CS287/CS287assignments/HW2/utils.lua")
dofile("/n/home09/ankitgupta/CS287/CS287assignments/HW2/models.lua")

function LogisticRegression(sparse_input, dense_input, training_output,
	validation_sparse_input, validation_dense_input, validation_output, 
	num_sparse_features, nclasses, minibatch_size, eta, num_epochs, lambda, 
	model_type, hidden_layers,  optimizer, word_embeddings, embedding_size, window_size, fixed_embeddings)

	local D_sparse_in, D_dense, D_output = num_sparse_features, dense_input:size(2), nclasses -- width of W_o, width of W_d, height of both W_o and W_d

	local model = nil
	local criterion = nil
	local embedding_layer = nil
	if model_type == "lr" then
		model, criterion = makeLogisticRegressionModel(D_sparse_in, D_dense, D_output, embedding_size, window_size)
		elseif model_type == "nnfig1" then
			model, criterion = makeNNmodel_figure1(D_sparse_in, D_dense, hidden_layers, D_output,embedding_size, window_size)
			elseif model_type == "nnpre" then
				model, criterion, embedding_layer = make_pretrained_NNmodel(D_sparse_in, D_dense, hidden_layers, D_output, window_size, word_embeddings)
			else
				assert(false)
			end

			print("Set up model")

	--local criterion = nn.ClassNLLCriterion()
	--local criterion = nn.MultiMarginCriterion()
	-- we can flatten (and then retrieve) all parameters (and gradParameters) of a module in the following way:
	local parameters, gradParameters = model:getParameters() -- N.B. getParameters() moves around memory, and should only be called once!
	print("Got params and grads")
	--print(training_output)
	--print("Counts of each of the classes in training set")
	--print(valueCounts(training_output))
	print("Starting Validation accuracy", getaccuracy(model, validation_sparse_input, validation_dense_input, validation_output))
	local num_minibatches = sparse_input:size(1)/minibatch_size

	for i = 1, num_epochs do
		print("L1 norm of params:", torch.abs(parameters):sum())

		for j = 1, sparse_input:size(1)-minibatch_size, minibatch_size do
			--print(j)
		    -- zero out our gradients
		    gradParameters:zero()
		    model:zeroGradParameters()

		    -- get the minibatch
		    sparse_vals = sparse_input:narrow(1, j, minibatch_size)
		    dense_vals = dense_input:narrow(1, j, minibatch_size)
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

				preds = model:forward({sparse_vals, dense_vals})
				loss = criterion:forward(preds, minibatch_outputs) --+ lambda*torch.norm(parameters,2)^2/2
				--print(loss)

				-- backprop
				dLdpreds = criterion:backward(preds, minibatch_outputs) -- gradients of loss wrt preds
				model:backward({sparse_vals, dense_vals}, dLdpreds)

				if fixed_embeddings then
					embedding_layer:zeroGradParameters()
				end

				-- return f and df/dX
				--return loss,gradParameters
				return loss,gradParameters --:add(parameters:clone():mul(lambda):div(num_minibatches))
			end
			--print(torch.abs(embedding_layer.weight):sum())
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
	print("Epoch "..i.." Validation accuracy:", getaccuracy(model, validation_sparse_input, validation_dense_input, validation_output))
end
return model
end
