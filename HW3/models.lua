require('nn')

--Returns an initialized MLP1 and NLL loss
--D_sparse_in is the number of words
--D_hidden is the dim of the hidden layer (tanhs)
--D_output is the number of words in this case
--embedding_size is the dimension of the input embedding
--window_size is the length of the context
function neuralNetwork(D_sparse_in, D_hidden, D_output, embedding_size, window_size)
	print("Making neural network model")

	local get_embeddings = nn.Sequential()

	get_embeddings:add(nn.LookupTable(D_sparse_in, embedding_size))
	get_embeddings:add(nn.View(-1):setNumInputDims(2))

	-- Apply a linear layer to those.
	get_embeddings:add(nn.Linear(embedding_size*window_size, D_hidden))

	local model = nn.Sequential()
	model:add(get_embeddings)
	model:add(nn.HardTanh())
	model:add(nn.Linear(D_hidden, D_output))
	model:add(nn.LogSoftMax())
	criterion = nn.ClassNLLCriterion()

	return model, criterion

end


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