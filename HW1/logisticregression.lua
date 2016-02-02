-- Only requirement allowed
require("hdf5")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')

-- Hyperparameters
-- ...
function unitTest()
	local x = torch.ones(3, 2)
	x[1][1] = 5
	x[2][1] = 3
	x[2][2] = 2
	x[3][1] = 2
	local Wt = torch.Tensor(4, 2)
	local b = torch.ones(1, 2)
	i=0; Wt:apply(function() i = i+1; return i end)
	Wt[4][2] = 0
	local output = validateModel(Wt:t(), b, x)
	print(output)

end

function sparseMultiply(A, B)
	-- A is a sparse tensor with 1-padding
	-- B is a dense tensor
	-- Matrix multiplication A*B in the straightforward way
	numRows = A:size(1)
	numCols = B:size(2)
	local output = torch.Tensor(numRows, numCols)
	for r = 1, numRows do
		for c = 1, numCols do
			local dotProd = 0
			-- dot product of row r in dense A and col c in B
			for j = 1, A:size(2) do		
				indexIntoB = A[r][j]-1
				if indexIntoB == 0 then
					break
				else
					dotProd = dotProd + B[indexIntoB][c]
				end
			end
			output[r][c] = dotProd
		end
	end
	return output
end


-- W and b are the weights to be trained. X is the sparse matrix representation of the input. Y is the classes
function validateModel(W, b, x, y)
    Ans = sparseMultiply(x, W:t())
    for r = 1, Ans:size(1) do
    	Ans[r]:add(b)
    end
    a, b = torch.max(Ans, 2)
    equality = torch.eq(b, y)
    --print(torch.sum(equality, 1)[1])

    score = equality:sum()/equality:size()[1]
    return score
end   
    



-- Given W which is a matrix (nclasses x nfeatures) and
--       X, which is a sparse matrix with 1-padding
-- Returns a Tensor (length nclasses) that contains softmax for each class
--     According to Murphy, when given W and x_i, this function returns the
--     vector [u_{i,1}   u_{i,2}   ...   u_{i,nclasses}]
function softmax(X, W)
	local W_size = W:size()
	local nclasses = W_size[1]
	local nfeatures = W_size[2]

	local Ans = sparseMultiply(X, W:t())
	--local Ans = torch.Tensor(nclasses)
	-- Perform the matrix multiplication
	--Ans:mv(W,x)

	-- exponentiate it
	Ans:exp()

	-- sum each row
	local row_sums = Ans:sum(2)

	-- normalize it
	for i = 1, X:size()[1] do
		--print(Ans[i], row_sums[i][1])
		Ans[i]:div(row_sums[i][1])
	end
	--Ans:div(Ans:sum())
	return Ans
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

-- Calculates the gradient of W
-- This is essentially an implementation of equation 8.40 from Murphy.
--      Will be using stochastic gradient descent with minibatches
-- Inputs:
--       W:           weights (nclasses x nfeatures)
--       Xs:          input features (sparse representation)
-- 		 Ys:          output classes (N x 1)
--       start_index: index of start of minibatch
--       end_index:   index of end of minibach
function gradient_W(W, Xs, Ys, start_index, end_index)
	local num_rows_wanted = end_index - start_index + 1

	-- Selects the num_rows_wanted rows that start at start_index
	local X = Xs:narrow(1,start_index,num_rows_wanted)

	-- Extract parameters
	local W_size = W:size()
	local nclasses = W_size[1]
	local nfeatures = W_size[2]

	-- Initiailize gradient 
	local W_grad = torch.zeros(nclasses, nfeatures)

	-- Calculate the softmax for each row
	-- softmax_res should be (num_rows_wanted X nclasses)
	local softmax_res = softmax(X, W)

	for n = 1, num_rows_wanted do
 		local class = Ys[n]
 		local softmax_val = softmax_res[n]

 		-- this is u_i - y_i
 		local diff = softmax_val - makeOneHot(class, nclasses)
 		local denseTensor = convertSparseToReal(Xs[n], nfeatures)
 		for i = 1, nclasses do
 			local tmp = torch.mul(denseTensor, diff[i])
 			W_grad[i] = W_grad[i] + tmp
 		end
	end
	return W_grad
end

function createCountsMatrix(training_input, training_output)
	print("     CreateCountsMatrix: Opened Data File")
	local f = hdf5.open(opt.datafile, 'r')
	local F = torch.zeros(nfeatures, nclasses)
	--local training_input = f:read(features_table):all():double()
	--local training_output = f:read(classes_table):all():double()
	print("     CreateCountsMatrix: Loaded training data")
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
	print("     CreateCountsMatrix: Calculated counts")
	return F
end

--[[
function naiveBayes(alpha)
	local f = hdf5.open(opt.datafile, 'r')
	local training_input = f:read('train_input'):all():double()
	local training_output = f:read('train_output'):all():double()

	local F = createCountsMatrix(training_input, training_output)

	-- Add a small offset for smoothing
	F = F + alpha

	-- Now, we column normalize the Tensor
	sum_of_each_col = torch.sum(F, 1)
	p_x_given_y = torch.Tensor(nfeatures, nclasses):zero()
	for n = 1, F:size()[1] do
		p_x_given_y[n] = torch.cdiv(F[n] , sum_of_each_col)
	end
	print("     NaiveBayes: Calculated p(x|y)")
	class_distribution = torch.zeros(nclasses)
	for n=1, training_output:size()[1] do
		class = training_output[n]
		class_distribution[class] = class_distribution[class] + 1
	end
	p_y = torch.div(class_distribution, torch.sum(class_distribution, 1)[1])
	print("     NaiveBayes: Calculated p(y)")

	local W = torch.log(p_x_given_y)
	local b = torch.log(p_y)
	local validation_input = f:read('valid_input'):all():double()
	local validation_output = f:read('valid_output'):all():long()
	print("     NaiveBayes: Initialized validation parameters")

    validation_accuracy = validateModel(W:t(), b, validation_input,validation_output)
	print("     NaiveBayes: Validation Accuracy:", validation_accuracy)

	return validation_accuracy	
end
--]]

function main() 
   -- Parse input params
   opt = cmd:parse(arg)
   local f = hdf5.open(opt.datafile, 'r')
   print("Opened datafile")
   nclasses = f:read('nclasses'):all():long()[1]
   nfeatures = f:read('nfeatures'):all():long()[1]
   print("Loaded size parameters")

   local W = torch.randn(nclasses, nfeatures)
   local b = torch.DoubleTensor(nclasses)
   local validation_input = f:read('valid_input'):all():double()
   local training_input = f:read('train_input'):all():double()
   local training_output = f:read('train_output'):all():double()
   print(validation_input:size())
   --print(validateModel(W, b, validation_input))
   --naiveBayes(1)
   print(torch.abs(gradient_W(W, training_input, training_output, 100, 200)):sum())
   --unitTest()
   -- Train.

   -- Test.
end

main()
--xtmp = torch.Tensor({{1,2,3},{1,2,3}})
--print(xtmp[2])
--print(xtmp[2][2])



