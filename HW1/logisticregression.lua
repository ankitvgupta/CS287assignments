dofile("utils.lua")

function logisticRegression(training_input, training_output, validation_input, validation_output, nfeatures, nclasses, minibatch_size, eta, lambda, num_epochs)
	--printv("Parameters are:", 2)
	--printv("Eta:", 2)
	--printv(eta, 2)
	--printv("Lambda:", 2)
	--printv(lambda, 2)
	--printv("Minibatch size:", 2)
	--printv(minibatch_size, 2)
	--printv("Number of Epochs", 2)
	--printv(num_epochs, 2)

	return SGD(training_input, training_output, validation_input, validation_output, nfeatures, nclasses, minibatch_size, eta, lambda, num_epochs)
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

function loss(W, b, Xs, Ys, lambda)
	local N = Xs:size()[1]
	local K = W:size()[1]
	local softmax_res = softmax(Xs, W, b)
	printv("	LOSS:Calculated softmax res", 3)
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

function crossEntropy(X, W, b, Y)
	local z = sparseMultiply(X, W:t())

	local numEntries = z:size()[1]
	local numClasses = z:size()[2]

	for i = 1, numEntries do
		z[i]:add(b)
	end
	-- z has numEntries rows and numClasses columns
	-- zc is a vector of length numEntries
	-- there must be a better way to do this!!!
	local zc = torch.Tensor(numEntries)
	for i = 1, numEntries do
		zc[i] = z[i][Y[i]]
	end
	-- log-sum-exp trick for log(sum(exp(zc')))
	-- M = max z_c'
	local M = torch.max(z, 2):t()[1]
	local total = torch.zeros(numEntries)
	-- now z has numClasses rows
	z = z:t()
	for c = 1, numClasses do
		z[c]:add(-M)
		total = total + z[c]:exp()
	end

	local crossEnt = -zc + total:log() + M
	return crossEnt:sum()

end

function crossEntropyLoss(W, b, Xs, Ys, lambda)
	return crossEntropy(Xs, W, b, Ys) + .5*lambda*torch.pow(W,2):sum()
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
function SGD(Xs, Ys, validation_input, validation_output, nfeatures, nclasses, minibatch_size, learning_rate, lambda, num_epochs)
	local testmode = false

	local N = Xs:size()[1]
	local W = torch.randn(nclasses, nfeatures)
	local b = torch.randn(nclasses)
	--local num_epochs = 10
	W:div(1)

	if testmode == true then
		N = 10000
		minibatch_size = 250
		num_epochs = 50
	end

	for rep = 1, num_epochs do
		-- Calculate the loss and validation accuracy
		printv("SGD: Loss is", 3)
		printv(loss(W, b, Xs, Ys, lambda), 3)
		printv("SGD: crossEntropyLoss is", 3)
		printv(crossEntropyLoss(W, b, Xs, Ys, lambda), 3)
		
		local validation_accuracy = validateLinearModel(W, b, validation_input,validation_output)
		
		printv("SGD: Validation Accuracy is:", 3)
		printv(validation_accuracy, 3)
		printv("SGD: Magnitude of W:", 3)
		printv(torch.abs(W):sum(), 3)
		printv("SGD: Magnitude of b:", 3)
		printv(torch.abs(b):sum(), 3)


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
				print("    Magnitude of W:",torch.abs(W):sum())
				print("    Magnitude of B_grad:", torch.abs(b_grad):sum())
				print("    Magnitude of b:", torch.abs(b):sum())
				print("\n")
			end
		end
	end
	return W, b
end
