dofile("utils.lua")

function hingeLogisticRegression(training_input, training_output, validation_input, validation_output, nfeatures, nclasses, minibatch_size, eta, lambda, num_epochs)
	printv("Parameters are:", 2)
	printv("Eta:", 2)
	printv(eta, 2)
	printv("Lambda:", 2)
	printv(lambda, 2)
	printv("Minibatch size:", 2)
	printv(minibatch_size, 2)
	printv("Number of Epochs", 2)
	printv(num_epochs, 2)

	return hingeSGD(training_input, training_output, validation_input, validation_output, nfeatures, nclasses, minibatch_size, eta, lambda, num_epochs)
end

function hingeLoss(W, b, Xs, Ys, lambda)
	local numEntries = Xs:size()[1]
	local numClasses = W:size()[1]

	local yp = sparseMultiply(Xs, W:t())
	local eb = torch.expand(b:view(numClasses, 1), numClasses, numEntries)
	yp = yp:add(eb)

	local globalMin = yp:min()

	-- Scores of true classes
	local ypc = torch.Tensor(numEntries)
	for i = 1, numEntries do
		ypc[i] = yp[i][Ys[i]]
		-- to allow for next max operation
		yp[i][Ys[i]] = globalMin
	end

	-- Scores of max false classes
	local ypp = yp:max(2):squeeze()

	local rlu = torch.ones(numEntries)-(ypc-ypp)
	rlu:apply(function(x) return (x<0) and 0 or x end)

	return rlu:sum() + lambda*torch.pow(W,2):sum()
end

function hingeGradient(W, b, Xs, Ys, start_index, end_index)
	local num_rows_wanted = end_index - start_index + 1

	-- Selects the num_rows_wanted rows that start at start_index
	local X = Xs:narrow(1,start_index,num_rows_wanted)
	local Y = Ys:narrow(1,start_index,num_rows_wanted)

	-- Extract parameters
	local W_size = W:size()
	local numClasses = W_size[1]
	local numFeatures = W_size[2]

	-- yp (y predicted) should be (num_rows_wanted X nclasses)
	local yp = sparseMultiply(X, W:t())
	local eb = torch.expand(b:view(numClasses, 1), numClasses, num_rows_wanted)
	yp = yp:add(eb)

	local globalMin = yp:min()

	-- Scores of true classes
	local ypc = torch.Tensor(num_rows_wanted)
	for i = 1, num_rows_wanted do
		ypc[i] = yp[i][Y[i]]
		-- to allow for next max operation
		yp[i][Y[i]] = globalMin
	end

	-- Scores of max false classes
	local ypp, Yps = yp:max(2)
	ypp = ypp:squeeze()
	Yps = Yps:squeeze()

	-- dL/dy
	local dLdy = torch.zeros(num_rows_wanted, numClasses)
	-- zero if ypc - ypp > 1
	for i = 1, num_rows_wanted do
		local c = Y[i]
		local cp = Yps[i]

		if ypc[i] - ypp[i] <= 1 then
			--print(c, cp, ypc[i], ypp[i])
			dLdy[i][c] = -1.0
			dLdy[i][cp] = 1.0
		end
	end

	-- Initialize gradient 
	local W_grad = torch.zeros(numClasses, numFeatures)
	local b_grad = torch.div(dLdy:sum(1), num_rows_wanted)

	--I think we can speed this up too
	for i=1, num_rows_wanted do
		for j=1, X:size()[2] do
			feat = X[i][j]-1
			if feat > 0 then
 				for c = 1, numClasses do
 					W_grad[c][feat] = W_grad[c][feat] +dLdy[i][c]/num_rows_wanted
 				end
 			end
 		end
	end

	return W_grad, b_grad
end

function hingeSGD(Xs, Ys, validation_input, validation_output, nfeatures, nclasses, minibatch_size, learning_rate, lambda, num_epochs)
	local testmode = false

	local N = Xs:size()[1]
	local W = torch.randn(nclasses, nfeatures)
	local b = torch.randn(nclasses)
	--local num_epochs = 10

	if testmode == true then
		N = 10000
		minibatch_size = 250
		num_epochs = 50
	end

	for rep = 1, num_epochs do
		-- Calculate the loss and validation accuracy
		printv("SGD: Hinge Loss is", 3)
		printv(hingeLoss(W, b, Xs, Ys, lambda), 3)

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
			local W_grad, b_grad = hingeGradient(W, b, Xs, Ys, start_index, end_index)
			W = W - (W_grad + torch.mul(W,lambda/N)):mul(learning_rate)
			b = b - (b_grad + torch.mul(b,lambda/N)):mul(learning_rate)
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
