-- This file contains the nn models used for training the various networks.
require('nn')

function makeLogisticRegressionModel(D_o, D_d, D_h)

	local par = nn.ParallelTable() -- takes a TABLE of inputs, applies i'th child to i'th input, and returns a table
	local sparse_multiply = nn.Sequential()
	sparse_multiply:add(nn.LookupTable(D_o, D_h))
	sparse_multiply:add(nn.Sum(1,2))

	par:add(sparse_multiply) -- first child
	par:add(nn.Linear(D_d, D_h)) -- second child
	
	local model = nn.Sequential()
	model:add(par)
	model:add(nn.CAddTable()) -- CAddTable adds its incoming tables

	model:add(nn.LogSoftMax())

	return model
end
