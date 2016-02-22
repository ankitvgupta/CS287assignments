-- inputs must be ints, 1 or greater
function valueCounts(t)
	max_val = torch.max(t)
	vals = torch.zeros(max_val)
	for i = 1, t:size(1) do
		vals[t[i]] = vals[t[i]] + 1
	end
	return vals
end


function printoptions(opt)
	print("Datafile:", opt.datafile, "Classifier:", opt.classifier, "Alpha:", opt.alpha, "Eta:", opt.eta, "Lambda:", opt.lambda, "Minibatch size:", opt.minibatch, "Num Epochs:", opt.epochs, "Optimizer:", opt.optimizer, "Hidden Layers:", opt.hiddenlayers, "Embedding size:", opt.embedding_size)
end


function createCountsMatrix(sparse_training_input, dense_training_input, training_output, n_sparse_features, nclasses)
	print("Num items", sparse_training_input:size(1), dense_training_input:size(1))
	local num_dense_features = dense_training_input:size(2)
	local F = torch.zeros(n_sparse_features + dense_training_input:size(2), nclasses):long()
	local train_size = sparse_training_input:size()
	for n = 1, train_size[1] do
		for j = 1, train_size[2] do
			feat = sparse_training_input[n][j]
			class = training_output[n]
			--print(feat,class)
			if feat > 0 then
				F[feat][class] = F[feat][class] + 1
			end
		end
	end

	-- Add the dense tensor to the last num_dense_features columns
	local last_rows = F:narrow(1, F:size()[1]-num_dense_features+1, num_dense_features)

	for class_val = 1,nclasses do
		-- Gets the indices of the rows that are in this class
		local indices_for_this_class = torch.range(1,training_output:size(1)):long()[torch.eq(training_output, class_val)]

		-- Gets the rows at those indicies and sums them
		local features_sum = dense_training_input:index(1, indices_for_this_class):sum(1):squeeze()

		-- Adds those to the right class in last_rows
		last_rows:select(2, class_val):add(features_sum)
	end

	return F
end


function testcountsmatrix()
	local sparse = torch.LongTensor({{4,9,3},{2,9,6},{3,8,1}})
	local num_sparse_features = 11

	local dense = torch.LongTensor({{1,0,1,1,0,1},{1,1,0,0,1,1},{1,1,0,0,0,0}})
	local output = torch.LongTensor{1,1,2}
	print(createCountsMatrix(sparse, dense, output, num_sparse_features, 2))

end

-- -- A is a sparse tensor with 1-padding
-- -- B is a dense tensor
-- -- Matrix multiplication A*B in the straightforward way
function sparseMultiply(A,B)
	local numRows = A:size()[1]
	local numCols = B:size()[2]

	local output = torch.Tensor(numRows, numCols)
	for r = 1, numRows do
		-- Grab the indicies that are not padding
		local indicies = (A[r])[A[r]:ge(1)]
		if (indicies:size():size() > 0) then
			output[r] = B:index(1, indicies):sum(1)
		end
	end
	return output
end

function getLinearModelPredictions(W, b, Xsparse, Xdense, num_sparse_features, num_dense_features)
	-- Split W components for the sparse features and dense features
	W_sparse = W:narrow(2, 1, num_sparse_features)
	W_dense = W:narrow(2, W:size(2) - num_dense_features + 1, num_dense_features)

	-- Sum the sparse and multiplication terms 
	local Ans = sparseMultiply(Xsparse, W_sparse:t())
	Ans:add(torch.mm(Xdense:double(), W_dense:t()))

    -- Add the bias
    for r = 1, Ans:size(1) do
    	Ans[r]:add(b)
    end


    local a, c = torch.max(Ans, 2)
    return c
end    


-- W and b are the weights to be trained. X is the sparse matrix representation of the input. Y is the classes
function validateLinearModel(W, b, x_sparse, x_dense, y, num_sparse_features, num_dense_features)
	local c = getLinearModelPredictions(W, b, x_sparse, x_dense, num_sparse_features, num_dense_features)
	print("min and max predicted class", torch.min(c), torch.max(c))
	equality = torch.eq(c, y)

	score = equality:sum()/equality:size()[1]
	return score
end
