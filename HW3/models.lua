require('nn')

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
--sample_probs is the same
function forwardPass(model, lookuptable, lookuptableparameters, lookuptablegradparameters, input_minibatch, output_minibatch, sample_indices, sample_probs)
	--dimension of z is minibatch_size x hidden_layer size
	local z = model:forward(input_minibatch)
	local K = sample_indices:size(1)
	local minibatch_size, hidden_size = z:size(1), z:size(2)
	--print(lookuptable.weight)

	for sample_i =1, sample_indices:size(1) do
		lookuptablegradparameters:zero()
		local idx = sample_indices[sample_i]
		local prob = sample_probs[sample_i]

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

		lookuptable:backward(torch.LongTensor{idx}, lookup_grad:sum(1))
		lookuptableparameters:add(torch.mul(lookuptablegradparameters,-.1))
	end

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

model, lookup = NCE(10, 2, 4, 2, 3)
input_batch = torch.LongTensor{{7, 5, 2},{1, 3, 4}}
output_batch = torch.Tensor{}
sample_indices = torch.LongTensor{1, 1, 1}
sample_probs = torch.Tensor{.1, .2, .3}
params, grads = lookup:getParameters()
forwardPass(model, lookup, params, grads, input_batch, output_batch, sample_indices, sample_probs)

function nn_predictall_and_subset(model, valid_input, valid_options)
	assert(valid_input:size(1) == valid_options:size(1))
	print("Starting predictions")
	local output_predictions = torch.zeros(valid_input:size(1), valid_options:size(2))
	print("Initialized output predictions tensor")
	local predictions = torch.exp(model:forward(valid_input))
	--print(predictions:sum(2))

	for i = 1, valid_input:size(1) do
		--if i % 100 == 0 then
		--	print("Iteration", i, "MemUsage", collectgarbage("count")*1024)
		--	collectgarbage()
		--end

		local values_wanted = predictions[i]:index(1, valid_options[i])
		--print(values_wanted)
		values_wanted:div(values_wanted:sum())
		output_predictions[i] = values_wanted
		--print(output_predictions[i]:sum())
	end
	return output_predictions
end
