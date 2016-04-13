require("nn")


function memm_model(nsparsefeatures, ndensefeatures, embeddingsize, D_win, D_out, hidden)
	print("Making MEMM Model")
	local parallel_table = nn.ParallelTable()
	local sparse_part = nn.Sequential()
	sparse_part:add(nn.LookupTable(nsparsefeatures, embeddingsize))
	sparse_part:add(nn.View(-1):setNumInputDims(2))
	print("D_win", D_win)

	if hidden > 0 then
		sparse_part:add(nn.Linear(embeddingsize*D_win, hidden))
	else
		sparse_part:add(nn.Linear(embeddingsize*D_win, D_out))
	end

	parallel_table:add(sparse_part)

	if hidden > 0 then
		parallel_table:add(nn.Linear(ndensefeatures, hidden))
	else
		parallel_table:add(nn.Linear(ndensefeatures, D_out))
	end

	local model = nn.Sequential()
	model:add(parallel_table)
	model:add(nn.CAddTable())

	if hidden > 0 then
		model:add(nn.Linear(hidden, D_out))
	end

	model:add(nn.LogSoftMax())

	criterion = nn.ClassNLLCriterion()
	return model,criterion
end

function train_memm(sparse_training_input, dense_training_input, training_output, 
						nsparsefeatures, ndensefeatures, nclasses, embeddingsize, num_epochs, minibatch_size, eta, optimizer, hidden)

	assert(sparse_training_input:size(1) == dense_training_input:size(1))
	assert(sparse_training_input:size(1) == training_output:size(1))

	--[[ 
		This first section of the code essentially adds the appropriate classes to the inputs.
		This is because the MEMM learns a function from (x_i, c_{i-1}) -> c_i
	--]]
	local ntrainingsamples = training_output:size(1) - 1
	-- Grab the first ntrainingsamples output_classes (all but last one), and add nsparsefeatures, which gives them a unique value
	local transformed_outputs = training_output:narrow(1, 1, ntrainingsamples) + nsparsefeatures

	-- Grab ntrainingsamples of the inputs (all but the first), and concatenate the classes.
	local sparse_input = torch.cat(sparse_training_input:narrow(1, 2, ntrainingsamples), transformed_outputs, 2)
	--print(sparse_input)
	-- Grab the associated rows of the others.
	local dense_input = dense_training_input:narrow(1, 2, ntrainingsamples)
	local output = training_output:narrow(1, 2, ntrainingsamples)

	-- print("nsparsefeatures", nsparsefeatures)
	-- print("nclasses", nclasses)

	--[[ 
		In this section, we train the above model.
	--]]

	local model, criterion = memm_model(nsparsefeatures+nclasses, ndensefeatures, embeddingsize, sparse_input:size(2), nclasses, hidden)
	print(model)
	local parameters, gradParameters = model:getParameters()
	for i = 1, num_epochs do
		print("Beginning epoch", i)

		for j = 1, ntrainingsamples-minibatch_size, minibatch_size do
			--print("J", j)

		    -- zero out our gradients
		    gradParameters:zero()
		    model:zeroGradParameters()

		   	minibatch_sparse_inputs = sparse_input:narrow(1, j, minibatch_size)
		   	minibatch_dense_inputs = dense_input:narrow(1, j, minibatch_size)
		    minibatch_outputs = output:narrow(1, j, minibatch_size)
		    -- print(torch.max(minibatch_sparse_inputs))
		    -- print(torch.max(minibatch_dense_inputs))
		    -- print(torch.max(minibatch_outputs))

		    --print(minibatch_dense_inputs)

		    -- Create a closure for optim
		    local feval = function(x)
				if x ~= parameters then
					parameters:copy(x)
				end

				-- reset gradients
				gradParameters:zero()

				preds = model:forward({minibatch_sparse_inputs,minibatch_dense_inputs})
				loss = criterion:forward(preds, minibatch_outputs) --+ lambda*torch.norm(parameters,2)^2/2
				if j == 1 then
					print("    ", loss)
				end

				-- backprop
				dLdpreds = criterion:backward(preds, minibatch_outputs) -- gradients of loss wrt preds
				--print(dLdpreds)
				model:backward({minibatch_sparse_inputs, minibatch_dense_inputs}, dLdpreds)


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


function make_predictor_function_memm(model, nsparsefeatures)

	local predictor = function(c_prev, x_i_sparse, x_i_dense)
		--print()
		local sparse = torch.zeros(x_i_sparse:size(1) + 1):long()
		sparse:narrow(1, 1, x_i_sparse:size(1)):copy(x_i_sparse)
		sparse[-1] = c_prev + nsparsefeatures
		--print(sparse)
		--print(sparse:reshape(1, sparse:size(1)))

		return torch.exp(model:forward({sparse:reshape(1, sparse:size(1)), x_i_dense:reshape(1, x_i_dense:size(1))})):squeeze()
	end
	return predictor

end