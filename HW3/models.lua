require('nn')

dofile('test.lua')
--Returns an initialized MLP1 and NLL loss
--D_sparse_in is the number of words
--D_hidden is the dim of the hidden layer (tanhs)
--D_output is the number of words in this case
--embedding_size is the dimension of the input embedding
--window_size is the length of the context
function neuralNetwork(D_sparse_in, D_hidden, D_output, embedding_size, window_size)
	print("Making neural network model")

	local model = nn.Sequential()
	model:add(nn.LookupTable(D_sparse_in, embedding_size))
	model:add(nn.View(-1):setNumInputDims(2))
	model:add(nn.Linear(embedding_size*window_size, D_hidden))
	model:add(nn.HardTanh())
	model:add(nn.Linear(D_hidden, D_output))
	model:add(nn.LogSoftMax())
	criterion = nn.ClassNLLCriterion()

	return model, criterion

end

--sample_indices a tensor of dimension 1xK
--probs is the distribution over all of the probabilities.
function forwardandBackwardPass(model, modelparams, modelgradparams, lookuptable, lookuptableparameters, lookuptablegradparameters, input_minibatch, output_minibatch, sample_indices, probs, eta)
	--dimension of z is minibatch_size x hidden_layer size
	model:zeroGradParameters()
	local z = model:forward(input_minibatch)
	local K = sample_indices:size(1)
	local minibatch_size, hidden_size = z:size(1), z:size(2)
	--print(lookuptable.weight)
	local total_gradient = torch.zeros(z:size())


	lookuptablegradparameters:zero()

	-- Grab the ML probabililities for each of the true words in the minibatch
	local prob = probs:index(1, output_minibatch)

	-- Grab the lookuptable rows for each of the true words. This is minibatchsize x hiddenlayer
	local lookuptable_rows = lookuptable:forward(output_minibatch)

	-- We now dot product the ith row of z with the ith row of lookuptable_rows (elementwise-mul and then sum left to right)
	-- The product will be minibatchsize x 1
	local dot_product = torch.cmul(z, lookuptable_rows):sum(2)
	local subtracted = dot_product - torch.log(torch.mul(prob, K))


	local sigmoid = nn.Sigmoid()
	-- Calculate the prediction for each of the inputs
	local predictions = sigmoid:forward(subtracted)
	criterion = nn.BCECriterion()
	-- Calculate the loss - note that these are all the correct ones, so the correct classes are all just 1
	local loss = criterion:forward(predictions, torch.ones(predictions:size(1)))
	--print(loss)

	-- Calculate the gradient
	dLdpreds = criterion:backward(predictions, torch.zeros(predictions:size(1))) -- gradients of loss wrt preds
	sigmoid:backward(subtracted, dLdpreds)
	
	local resizedGrad = (sigmoid.gradInput):view(minibatch_size, 1)
	-- Update the lookuptable 
	local lookup_grad = torch.cmul(z, resizedGrad:expand(minibatch_size, hidden_size))

	local overall_grad = torch.cmul(lookuptable_rows, resizedGrad:expand(minibatch_size, hidden_size))
	-- Add that to the gradient to be passed back to the model
	total_gradient:add(lookup_grad)
	--print(lookup_grad)
	lookuptable:backward(output_minibatch, lookup_grad)
	lookuptableparameters:add(torch.mul(lookuptablegradparameters,-1*eta))


	-- do the sampled cases
	for sample_i =1, sample_indices:size(1) do
		lookuptablegradparameters:zero()
		local idx = sample_indices[sample_i]
		local prob = probs[idx]

		local lookuptable_row = lookuptable:forward(torch.LongTensor{idx}):squeeze()

		-- minibatchsize x 1
		local dot_product = torch.mv(z, lookuptable_row)
		local subtracted = dot_product - torch.log(K*prob)
		local sigmoid = nn.Sigmoid()
		local predictions = sigmoid:forward(subtracted)

		criterion = nn.BCECriterion()
		local loss = criterion:forward(predictions, torch.zeros(predictions:size(1)))

		dLdpreds = criterion:backward(predictions, torch.zeros(predictions:size(1))) -- gradients of loss wrt preds
		sigmoid:backward(subtracted, dLdpreds)

		local resizedGrad = (sigmoid.gradInput):view(minibatch_size, 1)
		local lookup_grad = torch.cmul(z, resizedGrad:expand(minibatch_size, hidden_size))

		--lookuptable_row:view(1, lookuptable_row:size(1)):expand(minibatch_size, hidden_size)

		local overall_grad = torch.cmul(lookuptable_row:view(1, lookuptable_row:size(1)):expand(minibatch_size, hidden_size), resizedGrad:expand(minibatch_size, hidden_size))

		total_gradient:add(lookup_grad)
		--print(lookup_grad)
		lookuptable:backward(torch.LongTensor{idx}, lookup_grad:sum(1))
		lookuptableparameters:add(torch.mul(lookuptablegradparameters,-1*eta))
	end

	model:backward(input_minibatch, total_gradient)
	modelparams:add(torch.mul(modelgradparams,-1*eta))

end


function NCE(D_sparse_in, D_hidden, D_output, embedding_size, window_size)
	print("Making NCE neural network model")

	local model = nn.Sequential()
	model:add(nn.LookupTable(D_sparse_in, embedding_size))
	model:add(nn.View(-1):setNumInputDims(2))
	model:add(nn.Linear(embedding_size*window_size, D_hidden))
	model:add(nn.Tanh())
	-- we have z_w and z_{s_i,k} now


	return model, nn.LookupTable(D_sparse_in, D_hidden)

end

-- add the lookuptable weights to the model 
function make_NCEPredict_model(model, lookuptable, D_hidden, D_output)
	local linear_layer = nn.Linear(D_hidden, D_output)
	model:add(linear_layer)
	print(linear_layer.weight:size())
	print(lookuptable.weight:size())
	linear_layer.weight = lookuptable.weight
	linear_layer.bias = torch.zeros(linear_layer.bias:size())
	model:add(nn.LogSoftMax())
	return model
end

-- model, lookup = NCE(10, 2, 10, 2, 3)
-- modelparams, modelgradparams = model:getParameters()
-- input_batch = torch.LongTensor{{7, 5, 2}, {5,4,9}, {1,8,6}}
-- output_batch = torch.LongTensor{6,1,4}
-- sample_indices = torch.LongTensor{2, 2, 2, 3, 3}
-- sample_probs = torch.Tensor{.15, .05, .23, 05, .001, .01, .2, .3, .0001, .00001}
-- lookupparams, lookupgrads = lookup:getParameters()
-- --print(modelparams)
-- forwardandBackwardPass3(model, modelparams, modelgradparams,lookup, lookupparams, lookupgrads, input_batch, output_batch, sample_indices, sample_probs, 1)
-- --print(modelparams)
-- NCEPredict(model, lookup, 2, 10)

function nn_predictall_and_subset(model, valid_input, valid_options)
	assert(valid_input:size(1) == valid_options:size(1))
	print("Starting predictions")
	local output_predictions = torch.zeros(valid_input:size(1), valid_options:size(2))
	print("Initialized output predictions tensor")
	local predictions = torch.exp(model:forward(valid_input))

	for i = 1, valid_input:size(1) do
		--if i % 100 == 0 then
		--	print("Iteration", i, "MemUsage", collectgarbage("count")*1024)
		--	collectgarbage()
		--end
		--print(valid_options[i])
		local values_wanted = predictions[i]:index(1, valid_options[i])
		--print(values_wanted)
		values_wanted:div(values_wanted:sum())
		output_predictions[i]:add(values_wanted)
		--print(output_predictions[i]:sum())
	end
	--print(output_predictions)
	return output_predictions
end
