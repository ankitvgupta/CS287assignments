-- Only requirement allowed
require("hdf5")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')
cmd:option('-eta', .5, 'Learning rate')
cmd:option('-lambda', 1, 'regularization penalty')
cmd:option('-minibatch', 500, 'Minibatch size')
cmd:option('-epochs', 50, 'Number of epochs of SGD')


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



function fastSparseMultiply(A,B)
	local numRows = A:size()[1]
	local numCols = B:size()[2]

	local output = torch.Tensor(numRows, numCols)
	for r = 1, numRows do
		-- Grab the indicies that are not padding

		local indicies = (A[r] - 1)[(A[r] - 1):ge(1)]
		--print(A[r])
		--print(indicies)
		if (indicies:size():size() > 0) then
			output[r] = B:index(1, indicies):sum(1)
		end
	end
	return output
end

function sparseMultiply(A, B)
	return fastSparseMultiply(A, B)
end
	-- A is a sparse tensor with 1-padding
	-- B is a dense tensor
	-- Matrix multiplication A*B in the straightforward way
	--[[
	numRows = A:size(1)
	numCols = B:size(2)
	local output = torch.Tensor(numRows, numCols)
	for r = 1, numRows do
		for c = 1, numCols do
			local dotProd = 0
			-- dot product of row r in dense A and col c in B
			for j = 1, A:size(2) do		
				indexIntoB = A[r][j]-1
				if indexIntoB > 0 then
					dotProd = dotProd + B[indexIntoB][c]
				elseif indexIntoB == 0 then
					break
				end
			end
			output[r][c] = dotProd
		end
	end
	return output
end
--]]

-- W and b are the weights to be trained. X is the sparse matrix representation of the input. Y is the classes
function validateModel(W, b, x, y)
    Ans = sparseMultiply(x, W:t())
    for r = 1, Ans:size(1) do
    	Ans[r]:add(b)
    end
    a, b = torch.max(Ans, 2)
    equality = torch.eq(b, y)
    --print(b)
    --print(torch.sum(equality, 1)[1])

    score = equality:sum()/equality:size()[1]
    return score
end   
    



-- Given W which is a matrix (nclasses x nfeatures) and
--       X, which is a sparse matrix with 1-padding
-- Returns a Tensor (length nclasses) that contains softmax for each class
--     According to Murphy, when given W and x_i, this function returns the
--     vector [u_{i,1}   u_{i,2}   ...   u_{i,nclasses}]
function softmax(X, W, b)
	local W_size = W:size()
	local nclasses = W_size[1]
	local nfeatures = W_size[2]

	local Ans = sparseMultiply(X, W:t())
	for i = 1, Ans:size()[1] do
		Ans[i]:add(b)
	end
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


function loss(W, b, Xs, Ys, lambda)
	local N = Xs:size()[1]
	local K = W:size()[1]
	local softmax_res = softmax(Xs, W, b)
	print("LOSS:Calculated softmax res")
	local total = 0.0
	for n = 1, N do
		--print(n)
		for k = 1, K do
			local t = (Ys[n] == k) and 1 or 0
			local y = softmax_res[n][k]
			total = total + t*math.log(y)
		end
	end
	return ((-1)*total) + .5*lambda*torch.pow(W,2):sum()
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
function gradient(W, b, Xs, Ys, start_index, end_index)
	local num_rows_wanted = end_index - start_index + 1

	-- Selects the num_rows_wanted rows that start at start_index
	local X = Xs:narrow(1,start_index,num_rows_wanted)
	local Y = Ys:narrow(1,start_index,num_rows_wanted)

	-- Extract parameters
	local W_size = W:size()
	local nclasses = W_size[1]
	local nfeatures = W_size[2]

	-- Initiailize gradient 
	local W_grad = torch.zeros(nclasses, nfeatures)
	local b_grad = torch.zeros(nclasses)

	-- Calculate the softmax for each row
	-- softmax_res should be (num_rows_wanted X nclasses)
	local softmax_res = softmax(X, W, b)

	for n = 1, num_rows_wanted do
 		local class = Y[n]
 		local softmax_val = softmax_res[n]

 		-- this is u_i - y_i
 		local diff = softmax_val - makeOneHot(class, nclasses)
 		local denseTensor = convertSparseToReal(Xs[n], nfeatures)
 		for i = 1, nclasses do
 			local tmp = torch.mul(denseTensor, diff[i])
 			W_grad[i] = W_grad[i] + tmp:div(num_rows_wanted)
 		end
 		b_grad = b_grad + torch.div(diff, num_rows_wanted)
	end

	return W_grad, b_grad
end

-- Implements stochastic gradient descent with minibatching
function SGD(Xs, Ys, minibatch_size, learning_rate, lambda, num_epochs)
	local testmode = false


	local N = Xs:size()[1]
	local W = torch.randn(nclasses, nfeatures)
	local b = torch.randn(nclasses)
	--local num_epochs = 10
	W:div(1000)

	if testmode == true then
		N = 10000
		minibatch_size = 250
		num_epochs = 50
	end

	local f = hdf5.open(opt.datafile, 'r')
	local validation_input = f:read('valid_input'):all():long()
	local validation_output = f:read('valid_output'):all():long()
    

	for rep = 1, num_epochs do
		-- Calculate the loss and validation accuracy
		print("SGD: Loss is", loss(W, b, Xs, Ys, lambda))
		local validation_accuracy = validateModel(W, b, validation_input,validation_output)
		print("SGD: Validation Accuracy is:", validation_accuracy)
		print("Magnitude of W:", torch.abs(W):sum())
		print("Magnitude of b:", torch.abs(b):sum())


		local counter = 0
		for index = 1, N, minibatch_size do
			counter = counter + 1
			local start_index = index
			-- don't let the end_index exceed N
			local end_index = math.min(start_index + minibatch_size - 1, N)
			local size = end_index - start_index + 1
			--print(start_index, end_index)
			local W_grad, b_grad = gradient(W, b, Xs, Ys, start_index, end_index)
			W = W - (W_grad + torch.mul(W,lambda/N)):mul(learning_rate)
			b = b - torch.mul(b_grad,learning_rate)
			if counter % 20 == 0 then
				print("    Current index:", index)
				print("    Magnitude of W_grad:", torch.abs(W_grad):sum())
				print("    Magnitude of W:", torch.abs(W):sum())
				print("    Magnitude of B_grad:", torch.abs(W_grad):sum())
				print("    Magnitude of b:", torch.abs(W):sum(), "\n")
			end
		end
	end
	return W, b
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

function main() 
   -- Parse input params
   opt = cmd:parse(arg)
   local f = hdf5.open(opt.datafile, 'r')
   local eta = opt.eta
   local lambda = opt.lambda
   local minibatch_size = opt.minibatch
   local num_epochs = opt.epochs
   print("Parameters are:", "Eta:", eta, "Lambda", lambda, "Minibatch size:", minibatch_size, "Number of Epochs", num_epochs)
   print("Opened datafile")
   nclasses = f:read('nclasses'):all():long()[1]
   nfeatures = f:read('nfeatures'):all():long()[1]
   print("Loaded size parameters")

   local W = torch.randn(nclasses, nfeatures)
   local b = torch.DoubleTensor(nclasses)
   local training_input = f:read('train_input'):all():long()
   local training_output = f:read('train_output'):all():long()
   --print(validateModel(W, b, validation_input))
   --naiveBayes(1)
   --print(loss(W, b, training_input, training_output))
   SGD(training_input, training_output, minibatch_size, eta, lambda, num_epochs)
end


-- checks sparseMultiply by using convertSparseToReal and then doing normal matrix multiply
function checkSparseMultiply()
	--xtmp = torch.Tensor({{2,3,1,1},{3,1,1,1}})
	xtmp = (torch.rand(100,50):mul(3):abs():round() + 2):long()
	print(xtmp)
	local numfeatures = 6
	newarray = torch.zeros(xtmp:size()[1], numfeatures)
	for i = 1, newarray:size()[1] do
		newarray[i] = convertSparseToReal(xtmp[i], numfeatures)
	end
	print(newarray)
	local B = torch.randn(numfeatures, 3)
	print(torch.mm(newarray, B))
	print(fastSparseMultiply(xtmp, B))
end

main()




