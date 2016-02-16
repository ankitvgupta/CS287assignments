require('nn')

dofile("utils.lua")
dofile("models.lua")
-- -- For odyssey
-- dofile("/n/home09/ankitgupta/CS287/CS287assignments/HW2/utils.lua")
-- dofile("/n/home09/ankitgupta/CS287/CS287assignments/HW2/models.lua")

function LogisticRegression(sparse_input, dense_input, training_output,
	                        validation_sparse_input, validation_dense_input, validation_output, 
	                        num_sparse_features, nclasses, minibatch_size, eta, num_epochs, lambda, model_type, hidden_layers, optimizer)

	print("Began logistic regression")
	local D_o, D_d, D_h = num_sparse_features, dense_input:size(2), nclasses -- width of W_o, width of W_d, height of both W_o and W_d
	print("Got size parameters", D_o, D_d, D_h)

	local model = nil
	if model_type == "lr" then
		model = makeLogisticRegressionModel(D_o, D_d, D_h)
	elseif model_type == "nnfig1" then
		model = makeNNmodel_figure1(D_o, D_d, hidden_layers, D_h)
	else
		assert(false)
	end

	print("Set up model")

	local criterion = nn.ClassNLLCriterion()
	print("Set up criterion")
	-- we can flatten (and then retrieve) all parameters (and gradParameters) of a module in the following way:
	local parameters, gradParameters = model:getParameters() -- N.B. getParameters() moves around memory, and should only be called once!
	print("Got params and grads")
	--print(training_output)
	print("Counts of each of the classes in training set")
	print(valueCounts(training_output))
	print("Starting Validation accuracy", getaccuracy(model, validation_sparse_input, validation_dense_input, validation_output))
	local num_minibatches = sparse_input:size(1)/minibatch_size

	for i = 1, num_epochs do
		print("L1 norm of params:", torch.abs(parameters):sum())

		for j = 1, sparse_input:size(1)-minibatch_size, minibatch_size do
			--print(j)
		    -- zero out our gradients
		    gradParameters:zero()
		    model:zeroGradParameters()

		    -- get the minibatch
		    sparse_vals = sparse_input:narrow(1, j, minibatch_size)
		    dense_vals = dense_input:narrow(1, j, minibatch_size)
		    minibatch_outputs = training_output:narrow(1, j, minibatch_size)

		    -- Create a closure for optim
			local feval = function(x)
				-- Inspired by this torch demo: https://github.com/andresy/torch-demos/blob/master/train-a-digit-classifier/train-on-mnist.lua
				-- get new parameters
				if x ~= parameters then
					parameters:copy(x)
				end
				-- reset gradients
				gradParameters:zero()

				preds = model:forward({sparse_vals, dense_vals})
				loss = criterion:forward(preds, minibatch_outputs) --+ lambda*torch.norm(parameters,2)^2/2
				--print(loss)

				-- backprop
				dLdpreds = criterion:backward(preds, minibatch_outputs) -- gradients of loss wrt preds
				model:backward({sparse_vals, dense_vals}, dLdpreds)

				-- return f and df/dX
				--return loss,gradParameters
				return loss,gradParameters --:add(parameters:clone():mul(lambda):div(num_minibatches))
	    	end
	    	if optimizer == "adagrad" then
		    	config =  {
		    	            learningRate = eta,
	        	            weightDecay = lambda,
	        	            learningRateDecay = 5e-7
	                       }
	        optim.adagrad(feval, parameters, config)
	    	elseif optimizer == "sgd" then
	    		config = {
	        		   learningRate = eta, 
	                 }
	            optim.sgd(feval, parameters, config)
	    	else 
	    		assert(false)
	    	end
	        

		end
		print("Epoch "..i.." Validation accuracy:", getaccuracy(model, validation_sparse_input, validation_dense_input, validation_output))
	end
	return model
end

-- -- Given W which is a matrix (nclasses x nfeatures) and
-- --       X, which is a sparse matrix with 1-padding
-- -- Returns a Tensor (length nclasses) that contains softmax for each class
-- --     According to Murphy, when given W and x_i, this function returns the
-- --     vector [u_{i,1}   u_{i,2}   ...   u_{i,nclasses}]
-- function softmax(X, W, b)
-- 	local W_size = W:size()
-- 	local nclasses = W_size[1]
-- 	local nfeatures = W_size[2]

-- 	local Ans = sparseMultiply(X, W:t())
-- 	for i = 1, Ans:size()[1] do
-- 		Ans[i]:add(b)
-- 	end

-- 	-- exponentiate it
-- 	Ans:exp()

-- 	-- sum each row
-- 	local row_sums = Ans:sum(2)

-- 	-- normalize it
-- 	for i = 1, X:size()[1] do
-- 		Ans[i]:div(row_sums[i][1])
-- 	end

-- 	return Ans
-- end

-- function crossEntropy(X, W, b, Y)
-- 	local z = sparseMultiply(X, W:t())

-- 	local numEntries = z:size()[1]
-- 	local numClasses = z:size()[2]

-- 	for i = 1, numEntries do
-- 		z[i]:add(b)
-- 	end
-- 	-- z has numEntries rows and numClasses columns
-- 	-- zc is a vector of length numEntries

-- 	local zc = torch.Tensor(numEntries)
-- 	for i = 1, numEntries do
-- 		zc[i] = z[i][Y[i]]
-- 	end
-- 	-- log-sum-exp trick for log(sum(exp(zc')))
-- 	-- M = max z_c'
-- 	local M = torch.max(z, 2):t()[1]
-- 	local total = torch.zeros(numEntries)
-- 	-- now z has numClasses rows
-- 	z = z:t()
-- 	for c = 1, numClasses do
-- 		z[c]:add(-M)
-- 		total = total + z[c]:exp()
-- 	end

-- 	local crossEnt = -zc + total:log() + M
-- 	return crossEnt:sum()

-- end

-- function crossEntropyLoss(W, b, Xs, Ys, lambda)
-- 	return crossEntropy(Xs, W, b, Ys) + .5*lambda*torch.pow(W,2):sum()
-- end

-- -- Calculates the gradient of W
-- -- This is essentially an implementation of equation 8.40 from Murphy.
-- --      Will be using stochastic gradient descent with minibatches
-- -- Inputs:
-- --       W:           weights (nclasses x nfeatures)
-- --       Xs:          input features (sparse representation)
-- -- 		 Ys:          output classes (N x 1)
-- --       start_index: index of start of minibatch
-- --       end_index:   index of end of minibach
-- function gradient(W, b, Xs, Ys, start_index, end_index)
-- 	local num_rows_wanted = end_index - start_index + 1

-- 	-- Selects the num_rows_wanted rows that start at start_index
-- 	local X = Xs:narrow(1,start_index,num_rows_wanted)
-- 	local Y = Ys:narrow(1,start_index,num_rows_wanted)

-- 	-- Extract parameters
-- 	local W_size = W:size()
-- 	local nclasses = W_size[1]
-- 	local nfeatures = W_size[2]

-- 	-- Initiailize gradient 
-- 	local W_grad = torch.zeros(nclasses, nfeatures)
-- 	local b_grad = torch.zeros(nclasses)

-- 	-- Calculate the softmax for each row
-- 	-- softmax_res should be (num_rows_wanted X nclasses)
-- 	local softmax_res = softmax(X, W, b)

-- 	for n = 1, num_rows_wanted do
--  		local class = Y[n]
--  		local softmax_val = softmax_res[n]

--  		-- this is u_i - y_i
--  		local diff = softmax_val - makeOneHot(class, nclasses)
--  		local denseTensor = convertSparseToReal(X[n], nfeatures)
--  		for i = 1, nclasses do
--  			local tmp = torch.mul(denseTensor, diff[i])
--  			W_grad[i] = W_grad[i] + tmp:div(num_rows_wanted)
--  		end
--  		b_grad = b_grad + torch.div(diff, num_rows_wanted)
-- 	end

-- 	return W_grad, b_grad
-- end

-- -- Implements stochastic gradient descent with minibatching
-- function SGD(Xs, Ys, validation_input, validation_output, nfeatures, nclasses, minibatch_size, learning_rate, lambda, num_epochs)
-- 	local testmode = false

-- 	local N = Xs:size()[1]
-- 	local W = torch.randn(nclasses, nfeatures)
-- 	local b = torch.randn(nclasses)
-- 	--local num_epochs = 10
-- 	W:div(1)

-- 	if testmode == true then
-- 		N = 10000
-- 		minibatch_size = 250
-- 		num_epochs = 50
-- 	end

-- 	for rep = 1, num_epochs do
-- 		-- Calculate the loss and validation accuracy
-- 		printv("SGD: crossEntropyLoss is", 3)
-- 		printv(crossEntropyLoss(W, b, Xs, Ys, lambda), 3)
		
-- 		local validation_accuracy = validateLinearModel(W, b, validation_input,validation_output)
		
-- 		printv("SGD: Validation Accuracy is:", 3)
-- 		printv(validation_accuracy, 3)
-- 		printv("SGD: Magnitude of W:", 3)
-- 		printv(torch.abs(W):sum(), 3)
-- 		printv("SGD: Magnitude of b:", 3)
-- 		printv(torch.abs(b):sum(), 3)


-- 		local counter = 0
-- 		for index = 1, N, minibatch_size do
-- 			counter = counter + 1
-- 			local start_index = index
-- 			-- don't let the end_index exceed N
-- 			local end_index = math.min(start_index + minibatch_size - 1, N)
-- 			local size = end_index - start_index + 1
-- 			--print(start_index, end_index)
-- 			local W_grad, b_grad = gradient(W, b, Xs, Ys, start_index, end_index)
-- 			W = W - (W_grad + torch.mul(W,lambda/N)):mul(learning_rate)
-- 			b = b - torch.mul(b_grad,learning_rate)
-- 			if counter % 20 == 0 then
-- 				print("    Current index:", index)
-- 				print("    Magnitude of W_grad:", torch.abs(W_grad):sum())
-- 				print("    Magnitude of W:",torch.abs(W):sum())
-- 				print("    Magnitude of B_grad:", torch.abs(b_grad):sum())
-- 				print("    Magnitude of b:", torch.abs(b):sum())
-- 				print("\n")
-- 			end
-- 		end
-- 	end
-- 	return W, b
-- end
