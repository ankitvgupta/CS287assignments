require("nn")


-- Returns the structured perceptron model without softmax
function strucured_perceptron_model(nsparsefeatures, ndensefeatures, embeddingsize, D_win, D_out)
	print("Making MEMM Model")
	local parallel_table = nn.ParallelTable()
	local sparse_part = nn.Sequential()
	sparse_part:add(nn.LookupTable(nsparsefeatures, embeddingsize))
	sparse_part:add(nn.View(-1):setNumInputDims(2))
	sparse_part:add(nn.Linear(embeddingsize*D_win, D_out))

	parallel_table:add(sparse_part)
	parallel_table:add(nn.Linear(ndensefeatures, D_out))

	local model = nn.Sequential()
	model:add(parallel_table)
	model:add(nn.CAddTable())

	return model
end

--[[
You should call this function whenever a mistake is made.


model: the model being trained
input_sparse: the sparse features for the current word
input_dense: The dense features for the current word
c_i: the True class for the current word
c_iprev: The true class for the previous word
c_istar: THe predicted class for the current word
C_istarprev: The predicted class for the previous word

--]]
function single_update(model, input_sparse, input_dense, c_i, c_iprev, c_istar, c_istarprev, nsparsefeatures)

	-- Create the true previous sparse_input and the predicted previous sparse input.
	-- These are created by appending the true/predicted class to the input.
	local sparse_true = torch.cat(input_sparse, torch.LongTensor{c_iprev + nsparsefeatures})
	local sparse_pred = torch.cat(input_sparse, torch.LongTensor{c_istarprev + nsparsefeatures})

	-- Make them into a batch. - one with c_{i-1}, and one with c_{i-1}^*
	local batch_sparse = torch.cat(sparse_true, sparse_pred, 2):t()
	-- The dense inputs are the same for both.
	local batch_dense = torch.cat(input_dense, input_dense, 2):t()

	-- Make sure the shapes make sense.
	assert(batch_sparse:size(1) == 2)
	assert(batch_dense:size(1) == 2)

	-- Calculate the prediction.
	local preds = model:forward({batch_sparse, batch_dense})
	-- Manually created a gradient.
	local grad = torch.zeros(preds:size())
	grad[1][c_i] = -1
	grad[2][c_istar] = 1

	-- Push the gradient backwards.
	model:backward({batch_sparse, batch_dense}, grad)
end

-- Sentences should be a 3D tensor, where i,j,k is the ith sentence, jth window, kth feature.
-- Outputs should be 2D sentence, where i,j is the class for the jth window in the ith sentence.
-- TODO: Get the input in the format specified.
-- TODO: Test this :)
function train_structured_perceptron(sentences_sparse, sentences_dense, outputs, numepochs, nclasses, start_class, nsparsefeatures, ndensefeatures, embeddingsize, D_win)

	local model = strucured_perceptron_model(nsparsefeatures, ndensefeatures, embeddingsize, D_win, nclasses)
	local predictor = make_predictor_function_memm(model, nsparsefeatures)
	local parameters, gradParameters = model:getParameters()

	for i = 1, numepochs do
		-- For each sentence
		for j = 1, sentences:size(1) do
			-- Determine the predicted sequence
			predicted_sequence = viterbi(sentences_sparse[i], predictor, nclasses, start_class, sentences_dense[i])

			-- Compare the predicted sequence to the true sequence. Call single_update whenever there is a discrepancy.
			-- TODO: Check if predicted_sequence needs to be cast to a LongTensor (if it isn't already) for this to work.
			for k = 2, predicted_sequence:size(1) do
				if predicted_sequence[k] ~= outputs[j][k] then
					single_update(model, sentences_sparse[j][k], sentences_dense[j][k], outputs[j][k], outputs[j][k-1], predicted_sequence[k-1], predicted_sequence[k])
				end
			end
			-- Update the parameters with learning rate 1, as stated in the spec.
			parameters:add(-1.0, gradParameters)
		end
	end
end

function make_predictor_function_strucperpcetron(model, nsparsefeatures)

	local predictor = function(c_prev, x_i_sparse, x_i_dense)
		--print()
		local sparse = torch.zeros(x_i_sparse:size(1) + 1):long()
		sparse:narrow(1, 1, x_i_sparse:size(1)):copy(x_i_sparse)
		sparse[-1] = c_prev + nsparsefeatures
		--print(sparse)

		return model:forward({sparse:reshape(1, sparse:size(1)), x_i_dense:reshape(1, x_i_dense:size(1))}):squeeze()
	end
	return predictor

end