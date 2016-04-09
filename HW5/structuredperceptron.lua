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


function single_update(model, input_sparse, input_dense, c_i, c_iprev, c_istar, c_istarprev, nsparsefeatures)

	local sparse_true = torch.cat(input_sparse, torch.LongTensor{c_iprev + nsparsefeatures})
	local sparse_pred = torch.cat(input_sparse, torch.LongTensor{c_istarprev + nsparsefeatures})

	-- Two sparse inputs - one with c_{i-1}, and one with c_{i-1}^*
	local batch_sparse = torch.cat(sparse_true, sparse_pred, 2):t()
	-- The dense inputs are the same for both.
	local batch_dense = torch.cat(input_dense, input_dense, 2):t()
	assert(batch_sparse:size(1) == 2)
	assert(batch_dense:size(1) == 2)

	local preds = model:forward({batch_sparse, batch_dense})

	local grad = torch.zeros(preds:size())
	grad[1][c_i] = -1
	grad[2][c_istar] = 1

	model:backward({batch_sparse, batch_dense}, grad)


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