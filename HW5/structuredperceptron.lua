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