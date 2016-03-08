
--sample_indices a tensor of dimension 1xK
--probs is the distribution over all of the probabilities.
function forwardandBackwardPass3(model, modelparams, modelgradparams, lookuptable, lookuptableparameters, lookuptablegradparameters, input_minibatch, output_minibatch, sample_indices, probs, eta, bias, biasparams, biasgradparams)
	--dimension of z is minibatch_size x hidden_layer size
	model:zeroGradParameters()
	lookuptable:zeroGradParameters()
	bias:zeroGradParameters()
	local tanh_result = model:forward(input_minibatch)



	local minibatch_size = input_minibatch:size(1)
	--print("Size", minibatch_size)

	local K = sample_indices:size(1)

	-- Determine which rows to pick from lookuptable (each row of rows_wanted correspond to the indicies wanted for that minibatch)
	local rows_wanted = torch.zeros(minibatch_size, K+1):long()
	rows_wanted:narrow(2, 2, K):add(sample_indices:view(1, K):expand(minibatch_size,K))
	rows_wanted:select(2,1):add(output_minibatch)

	local lookuptable_rows = lookuptable:forward(rows_wanted)
	local bias_rows = bias:forward(rows_wanted)

	
	-- Set whether each of these are true word or sampled or not (all are sampled, except the first is true)
	local classifications = torch.zeros(minibatch_size, K+1)
	classifications:select(2,1):fill(1)


	-- Get the p_ML for each of these words
	pmls = torch.zeros(minibatch_size, K+1)
	pmls:select(2,1):add(probs:index(1, output_minibatch))
	pmls:narrow(2, 2, K):add(probs:index(1, sample_indices):view(1, K):expand(minibatch_size,K))

	local z = torch.zeros(minibatch_size, K+1)
	for i = 1, minibatch_size do
		--print(bias_rows[i]:t())
		z[i] = torch.mm(tanh_result[i]:view(1, tanh_result:size(2)), lookuptable_rows[i]:t()) + bias_rows[i]:t()
	end



	local subtracted = z - torch.log(torch.mul(pmls, K))
	local sigmoid = nn.Sigmoid()
	-- Calculate the prediction for each of the inputs
	local predictions = sigmoid:forward(subtracted)
	criterion = nn.BCECriterion()

	-- Calculate the loss
	local loss = criterion:forward(predictions, classifications)

	-- Calculate the gradient
	local dLdpreds = criterion:backward(predictions, classifications) -- gradients of loss wrt preds
	local sigmoid_grad = sigmoid:backward(subtracted, dLdpreds)

	local lookup_grad = torch.zeros(lookuptable_rows:size())
	for i = 1, minibatch_size do
		lookup_grad[i] = torch.mm(sigmoid_grad[i]:view(K+1, 1), tanh_result[i]:view(1, tanh_result:size(2)))
	end

	local bias_grad = torch.zeros(bias_rows:size())
	for i = 1, minibatch_size do
		bias_grad[i] = sigmoid_grad[i]
	end



	local model_grad = torch.zeros(tanh_result:size())
	--print(lookup_grad)
	for i = 1, minibatch_size do
		--print(sigmoid_grad[i], lookuptable_rows[i])
		model_grad[i] = torch.mm(sigmoid_grad[i]:view(1, K+1), lookuptable_rows[i])
		--print(tanh_result[i])
		--z[i] = torch.mm(tanh_result[i]:view(1, tanh_result:size(2)), lookuptable_rows[i]:t())
	end


	lookuptable:backward(rows_wanted, lookup_grad)
	--lookuptableparameters:add(torch.mul(lookuptablegradparameters,-1*eta))
	bias:backward(rows_wanted, bias_grad)
	model:backward(input_minibatch, model_grad)

	lookuptableparameters:add(-eta, lookuptablegradparameters)
	biasparams:add(-eta, biasgradparams)
	--biasparams:add(torch.mul(biasgradparams,-1*eta))

	--print(model)
	modelparams:add(-eta, modelgradparams)
	--modelparams:add(torch.mul(modelgradparams,-1*eta))


	--local lookup_grad = torch.cmul(z, sigmoid_grad:expand(minibatch_size, hidden_size))


end

	

