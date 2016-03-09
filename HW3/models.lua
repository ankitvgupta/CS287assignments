require('nn')

dofile(_G.path.."test.lua")

--dofile('test.lua')
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


-- Create an NCE model
function NCE(D_sparse_in, D_hidden, D_output, embedding_size, window_size)
	print("Making NCE neural network model")

	local model = nn.Sequential()
	local embedding = nn.LookupTable(D_sparse_in, embedding_size)
	model:add(embedding)
	model:add(nn.View(-1):setNumInputDims(2))
	model:add(nn.Linear(embedding_size*window_size, D_hidden))
	model:add(nn.Tanh())
	-- we have z_w and z_{s_i,k} now


	return model, embedding, nn.LookupTable(D_sparse_in, D_hidden), nn.LookupTable(D_sparse_in, 1)

end


-- Returns the global cross-entropy of a standard NNLM.
function NNLM_CrossEntropy(model, to_predict_input, true_outputs)

	model:zeroGradParameters()
	local crit = nn.ClassNLLCriterion()
	crit.sizeAverage = false
	local total = 0
	for i = 1, to_predict_input:size(1) - 100, 100 do
		local preds = model:forward(to_predict_input:narrow(1, i, 100))
		local cross_entropy_loss = crit:forward(preds, true_outputs:narrow(1, i, 100))
		total = total + cross_entropy_loss
	end
	return total/to_predict_input:size(1)
end

-- Returns the global cross-entropy of a NCE model.
function NCE_predictions2(model, lookuptable, bias, to_predict_input, true_outputs, D_hidden, D_output)

	--model:zeroGradParameters()
	--lookuptable:zeroGradParameters()
	--bias:zeroGradParameters()
	--print("NumValidationInputs", to_predict_input:size(1))

	local prediction_err = nn.Sequential()
	local linear_layer = nn.Linear(D_hidden, D_output)
	-- print(linear_layer.weight:size())
	-- print(lookuptable.weight:size())
	-- print(linear_layer.bias:size())
	-- print(bias.weight:squeeze():size())

	linear_layer.weight = lookuptable.weight
	linear_layer.bias = bias.weight:squeeze()
	prediction_err:add(linear_layer)
	prediction_err:add(nn.LogSoftMax())
	
	--print(linear_layer.weight - lookuptable.weight)
	--print(bias.weight)
	local crit = nn.ClassNLLCriterion()
	crit.sizeAverage = false
	local total = 0
	for i = 1, to_predict_input:size(1) - 100, 100 do
		local tanh_result = model:forward(to_predict_input:narrow(1, i, 100))
		local preds = prediction_err:forward(tanh_result)
		local cross_entropy_loss = crit:forward(preds, true_outputs:narrow(1, i, 100))
		total = total + cross_entropy_loss
	end
	return total/to_predict_input:size(1)
end

-- For each validation input, returns a distribution over the options.
-- This can be used to calculate the cross-entropy over the options.
function NCE_predictions(model, lookuptable, bias, to_predict_input, to_predict_options, probs)

	model:zeroGradParameters()
	lookuptable:zeroGradParameters()
	bias:zeroGradParameters()

	local tanh_result = model:forward(to_predict_input)
	local minibatch_size = to_predict_input:size(1)
	local K = to_predict_options:size(2)

	-- Determine which rows to pick from lookuptable (each row of rows_wanted correspond to the indicies wanted for that minibatch)
	local rows_wanted = to_predict_options

	local lookuptable_rows = lookuptable:forward(rows_wanted)
	local bias_rows = bias:forward(rows_wanted)

	local z = torch.zeros(minibatch_size, K)
	for i = 1, minibatch_size do
		--print(bias_rows[i]:t())
		z[i] = torch.mm(tanh_result[i]:view(1, tanh_result:size(2)), lookuptable_rows[i]:t()) + bias_rows[i]:t()
	end

	predictions = nn:LogSoftMax():forward(z)
	return predictions
end

function getNCEStats(model, lookup, bias, valid_input, valid_options, valid_true_outs, sample_probs)
    local predictions = NCE_predictions(model, lookup, bias, valid_input, valid_options, sample_probs)
    --print(predictions:sum(2))
    local cross_ent = cross_entropy_loss(valid_true_outs, predictions, valid_options)
    return get_result_accuracy(predictions, valid_input, valid_options, valid_true_outs), cross_ent, torch.exp(cross_ent)
end
-- model, lookup, bias = NCE(10, 2, 10, 2, 3)
-- modelparams, modelgradparams = model:getParameters()
-- biasparams, biasgrad = model:getParameters()
-- input_batch = torch.LongTensor{{7, 5, 2}, {5,4,9}, {1,8,6}}
-- output_batch = torch.LongTensor{6,1,4}
-- sample_indices = torch.LongTensor{2, 2, 2, 3, 3}
-- sample_probs = torch.Tensor{.15, .05, .23, 05, .001, .01, .2, .3, .0001, .00001}
-- lookupparams, lookupgrads = lookup:getParameters()
-- --print(modelparams)
-- forwardandBackwardPass3(model, modelparams, modelgradparams,lookup, lookupparams, lookupgrads, input_batch, output_batch, sample_indices, sample_probs, 1, bias, biasparams, biasgrad)
-- --print(modelparams)
-- NCE_predictions(model, lookup, bias, torch.LongTensor{{7, 5, 2}, {5,4,9}}, torch.LongTensor{{1,2,3,4}, {1,2,3,4}} )

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
	return torch.log(output_predictions)
end
