VERBOSITY = 3

function printv(s, verbosity)
	if VERBOSITY >= verbosity then
		print(s)
	end
end

function createCountsMatrix(training_input, training_output, nfeatures, nclasses)

	local F = torch.zeros(nfeatures, nclasses)
	--local training_input = f:read(features_table):all():double()
	--local training_output = f:read(classes_table):all():double()
	printv("     CreateCountsMatrix: Loaded training data", 3)
	local train_size = training_input:size()
	for n = 1, train_size[1] do
		for j = 1, train_size[2] do
			feat = training_input[n][j] - 1
			class = training_output[n]
			if feat > 0 then
				F[feat][class] = F[feat][class] + 1
			end
		end
	end
	printv("     CreateCountsMatrix: Calculated counts", 3)
	return F
end

-- returns a one-hot tensor of size n with a 1 the ith place.
--    Note that this is 1-indexed
function makeOneHot(i, n)
	local tmp_tensor = torch.zeros(n)
	tmp_tensor[i] = 1
	return tmp_tensor
end

-- convert from sparse 1-padded 1-dimensional tensor to real 1-dimensional tensor
function convertSparseToReal(sparse, numFeatures)
	local res = torch.zeros(numFeatures):double()
	for i = 1, sparse:size()[1] do
		local ind = sparse[i] - 1
		if ind > 0 then
			res[ind] = res[ind] + 1
		end
	end
	return res
end


-- A is a sparse tensor with 1-padding
-- B is a dense tensor
-- Matrix multiplication A*B in the straightforward way
function sparseMultiply(A,B)
	local numRows = A:size()[1]
	local numCols = B:size()[2]

	local output = torch.Tensor(numRows, numCols)
	for r = 1, numRows do
		-- Grab the indicies that are not padding
		local indicies = (A[r] - 1)[(A[r] - 1):ge(1)]
		if (indicies:size():size() > 0) then
			output[r] = B:index(1, indicies):sum(1)
		end
	end
	return output
end

-- W and b are the weights to be trained. X is the sparse matrix representation of the input. Y is the classes
function validateLinearModel(W, b, x, y)
    local Ans = sparseMultiply(x, W)
    for r = 1, Ans:size(1) do
    	Ans[r]:add(b)
    end
    a, c = torch.max(Ans, 2)
    equality = torch.eq(c, y)

    score = equality:sum()/equality:size()[1]
    return score
end   

-- Returns another tensor with only the rows of X that have sentences of length minlength

function removeSmallSentences(X, Y, minlength)
    local desired_row_mask = torch.ge(torch.ge(X, 2):sum(2), minlength)
    local desired_rows = torch.range(1, X:size()[1]):long()[desired_row_mask]
    return X:index(1, desired_rows), Y:index(1, desired_rows)
end
