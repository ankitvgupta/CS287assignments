-- This file contains the nn models used for training the various networks.
require('nn')
--require('cudnn')
-- This function makes the logistic regression model
function makeLogisticRegressionModel(D_sparse_in, D_dense, D_output,  embedding_size, window_size)
	print("Making lr model")

	local par = nn.ParallelTable()
	local sparse_multiply = nn.Sequential()
	sparse_multiply:add(nn.LookupTable(D_sparse_in, D_output))
	sparse_multiply:add(nn.Sum(1,2))

	par:add(sparse_multiply) -- first child
	par:add(nn.Linear(D_dense, D_output)) -- second child
	
	local model = nn.Sequential()
	model:add(par)
	model:add(nn.CAddTable())

	model:add(nn.LogSoftMax())
	criterion = nn.ClassNLLCriterion()
	return model, criterion
end

-- This function builds the neural network used in the paper
function makeNNmodel_figure1(D_sparse_in, D_dense, D_hidden, D_output,  embedding_size, window_size)
	print("Making neural network model")

	local par = nn.ParallelTable()
	local get_embeddings = nn.Sequential()

	get_embeddings:add(nn.LookupTable(D_sparse_in, embedding_size))
	get_embeddings:add(nn.View(-1):setNumInputDims(2))

	-- Apply a linear layer to those.
	get_embeddings:add(nn.Linear(embedding_size*window_size, D_hidden))

	par:add(get_embeddings) -- first child
	par:add(nn.Linear(D_dense, D_hidden)) -- second child
	
	local model = nn.Sequential()
	model:add(par)
	model:add(nn.CAddTable())
	model:add(nn.HardTanh())
	model:add(nn.Linear(D_hidden, D_output))
	model:add(nn.LogSoftMax())
	criterion = nn.ClassNLLCriterion()


	return model, criterion

end

function make_pretrained_NNmodel(D_sparse_in, D_dense, D_hidden, D_output, window_size, word_embeddings)
	print("Making neural network model 2")

	local par = nn.ParallelTable()
	local get_embeddings = nn.Sequential()
	local embedding_size = word_embeddings:size(2)

	local lookup = nn.LookupTable(word_embeddings:size())
	lookup.weight = word_embeddings

	get_embeddings:add(lookup)
	get_embeddings:add(nn.View(-1):setNumInputDims(2))
	get_embeddings:add(nn.Linear(embedding_size*window_size, D_hidden))


	par:add(get_embeddings) -- first child
	par:add(nn.Linear(D_dense, D_hidden)) -- second child
	
	local model = nn.Sequential()
	model:add(par)
	model:add(nn.CAddTable())
	model:add(nn.HardTanh())
	model:add(nn.Linear(D_hidden, D_output)) -- second child
	model:add(nn.LogSoftMax())
	criterion = nn.ClassNLLCriterion()
	local usegpu = true
	--if usegpu then
	--	cudnn.convert(model, cudnn)
	--end
	return model, criterion, lookup

end

function getaccuracy(model, validation_sparse_input, validation_dense_input, validation_output)
	scores = model:forward({validation_sparse_input, validation_dense_input})
	local _, class_preds = torch.max(scores, 2)
	local equality = torch.eq(class_preds, validation_output)
	local score = equality:sum()/equality:size()[1]
	return score
end
