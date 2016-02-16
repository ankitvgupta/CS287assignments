-- This file contains the nn models used for training the various networks.
require('nn')

-- This function makes the logistic regression model
function makeLogisticRegressionModel(D_sparse_in, D_dense, D_output,  embedding_size, window_size)
	print("Making lr model")

	local par = nn.ParallelTable() -- takes a TABLE of inputs, applies i'th child to i'th input, and returns a table
	local sparse_multiply = nn.Sequential()
	sparse_multiply:add(nn.LookupTable(D_sparse_in, D_output))
	sparse_multiply:add(nn.Sum(1,2))

	--sparse_multiply:add(nn.LookupTable(D_sparse_in, embedding_size))
	-- Flatten those features into a single vector  
	--sparse_multiply:add(nn.View(-1):setNumInputDims(2))
	-- Apply a linear layer to those.
	-- THIS ASSUMES D_WIN - 3 -- TODO: MAKE THAT BE AN ARG
	--sparse_multiply:add(nn.Linear(embedding_size*window_size, D_output))

	par:add(sparse_multiply) -- first child
	par:add(nn.Linear(D_dense, D_output)) -- second child
	
	local model = nn.Sequential()
	model:add(par)
	model:add(nn.CAddTable()) -- CAddTable adds its incoming tables

	model:add(nn.LogSoftMax())
	criterion = nn.ClassNLLCriterion()
	return model, criterion
end

-- This function builds the neural network used in the paper

function makeNNmodel_figure1(D_sparse_in, D_dense, D_hidden, D_output,  embedding_size, window_size)
	print("Making neural network model")

	local par = nn.ParallelTable() -- takes a TABLE of inputs, applies i'th child to i'th input, and returns a table
	local get_embeddings = nn.Sequential()
	--sparse_multiply:add(nn.LookupTable(D_sparse_in, D_hidden))
	--sparse_multiply:add(nn.Sum(1,2))

	-- I THINK this is now correct.
	-- Get the word embeddings for each of the words
	get_embeddings:add(nn.LookupTable(D_sparse_in, embedding_size))
	-- Flatten those features into a single vector  
	get_embeddings:add(nn.View(-1):setNumInputDims(2))
	-- Apply a linear layer to those.
	-- THIS ASSUMES D_WIN - 3 -- TODO: MAKE THAT BE AN ARG
	get_embeddings:add(nn.Linear(embedding_size*window_size, D_hidden))


	par:add(get_embeddings) -- first child
	par:add(nn.Linear(D_dense, D_hidden)) -- second child
	
	local model = nn.Sequential()
	model:add(par)
	model:add(nn.CAddTable()) -- CAddTable adds its incoming tables
	model:add(nn.HardTanh())
	model:add(nn.Linear(D_hidden, D_output)) -- second child
	model:add(nn.LogSoftMax())
	criterion = nn.ClassNLLCriterion()

	local x = torch.LongTensor({{45,12,13},{17,3,4}})
	local x2 = torch.randn(2, D_dense)
	print("forward test")
	print(par:forward({x, x2}))


	return model, criterion

end

-- TODO: Change this to use the new approach of having one embedding per word!
function make_pretrained_NNmodel(D_o, D_d, D_hidden, D_output, D_win, word_embeddings)
	print("Making neural network model pretrained")

	local par = nn.ParallelTable() -- takes a TABLE of inputs, applies i'th child to i'th input, and returns a table
	local sparse_multiply = nn.Sequential()
	local lookup = nn.LookupTable(D_o, D_hidden)

	local catbeddings = word_embeddings

	for i=1, 4 do
		catbeddings = catbeddings:cat(word_embeddings, 1)
	end

	lookup.weight = catbeddings

	sparse_multiply:add(lookup)
	sparse_multiply:add(nn.Sum(1,2))

	par:add(sparse_multiply) -- first child
	par:add(nn.Linear(D_d, D_hidden)) -- second child
	
	local model = nn.Sequential()
	model:add(par)
	model:add(nn.CAddTable()) -- CAddTable adds its incoming tables
	model:add(nn.HardTanh())
	model:add(nn.Linear(D_hidden, D_output)) -- second child
	model:add(nn.LogSoftMax())
	local criterion = nn.ClassNLLCriterion()

	return model, criterion

end

function getaccuracy(model, validation_sparse_input, validation_dense_input, validation_output)
	scores = model:forward({validation_sparse_input, validation_dense_input})
	local _, class_preds = torch.max(scores, 2)
	--print(valueCounts(class_preds:squeeze()))
	local equality = torch.eq(class_preds, validation_output)
	local score = equality:sum()/equality:size()[1]
	return score
end