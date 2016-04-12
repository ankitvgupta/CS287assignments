require("nn")


-- Returns the structured perceptron model without softmax
function structured_perceptron_model(nsparsefeatures, ndensefeatures, embeddingsize, D_win, D_out)
	print("Making Structured Perceptron Model")
	local parallel_table = nn.ParallelTable()
	local sparse_part = nn.Sequential()
	sparse_part:add(nn.LookupTable(nsparsefeatures, embeddingsize))
	sparse_part:add(nn.View(-1):setNumInputDims(2))
	-- print("Dwin", D_win)
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

	-- print(preds[1])

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
function train_structured_perceptron(sentences_sparse, sentences_dense, outputs, numepochs, nclasses, start_class, end_class, nsparsefeatures, ndensefeatures, embeddingsize, eta)

	local model = structured_perceptron_model(nsparsefeatures+nclasses, ndensefeatures, embeddingsize, sentences_sparse[1]:size(2) + 1, nclasses)
	local predictor = make_predictor_function_strucperceptron(model, nsparsefeatures, end_class, nclasses, start_class)
	local parameters, gradParameters = model:getParameters()

	for i = 1, numepochs do
		print("Starting epoch...", i)
		local total_correct = 0
		local total = 0

		-- For each sentence
		for j = 1, #sentences_sparse do
			-- Determine the predicted sequence
			predicted_sequence = viterbi(sentences_sparse[j], predictor, nclasses, start_class, sentences_dense[j]):long()

			assert(predicted_sequence:size(1) == outputs[j]:size(1))
			--print("Percent same", torch.eq(predicted_sequence, outputs[j]):sum()/predicted_sequence:size(1))
			total_correct = total_correct + torch.eq(predicted_sequence, outputs[j]):sum()
			total = total + predicted_sequence:size(1)
			--print(predicted_sequence)

			-- Compare the predicted sequence to the true sequence. Call single_update whenever there is a discrepancy.
			-- TODO: Check if predicted_sequence needs to be cast to a LongTensor (if it isn't already) for this to work.
			-- print("Comparison")
			-- print(predicted_sequence)
			-- print(outputs[j])
			for k = 2, predicted_sequence:size(1) do
				if predicted_sequence[k] ~= outputs[j][k] then
					single_update(model, sentences_sparse[j][k], sentences_dense[j][k], outputs[j][k], outputs[j][k-1], predicted_sequence[k], predicted_sequence[k-1], nsparsefeatures)
				end
			end
			-- Update the parameters with learning rate 1, as stated in the spec.
			--print(torch.abs(gradParameters):sum())
			parameters:add(-eta, gradParameters)
			gradParameters:zero()
			model:zeroGradParameters()
		end
		print("   Epoch "..i.." Percent correct",total_correct/total )
	end

	return model, predictor
end

function make_predictor_function_strucperceptron(model, nsparsefeatures, end_class, nclasses, begin_class)

	local predictor = function(c_prev, x_i_sparse, x_i_dense)
		--print()
		local sparse = torch.zeros(x_i_sparse:size(1) + 1):long()
		sparse:narrow(1, 1, x_i_sparse:size(1)):copy(x_i_sparse)
		sparse[-1] = c_prev + nsparsefeatures
		--print(sparse)
		-- print("hi", c_prev, sparse, x_i_dense)
		-- print(sparse:reshape(1, sparse:size(1)))
		-- print(x_i_dense:reshape(1, x_i_dense:size(1)))
	
		local x = model:forward({sparse:reshape(1, sparse:size(1)), x_i_dense:reshape(1, x_i_dense:size(1))}):squeeze()
	
		if c_prev == end_class then
			x:zero()
			x[begin_class] = 1000
		end
		--print(x)

		return x
	end

	return predictor

end