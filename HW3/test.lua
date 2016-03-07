
--sample_indices a tensor of dimension 1xK
--probs is the distribution over all of the probabilities.
function forwardandBackwardPass3(model, modelparams, modelgradparams, lookuptable, lookuptableparameters, lookuptablegradparameters, input_minibatch, output_minibatch, sample_indices, probs, eta)
	--dimension of z is minibatch_size x hidden_layer size
	model:zeroGradParameters()
	local tanh_result = model:forward(input_minibatch)
	--print(tanh_result)
	--tanh_result = tanh_result:view(1, tanh_result:size(1))


	local minibatch_size = input_minibatch:size(1)
	--print("Size", minibatch_size)

	local K = sample_indices:size(1)

	-- Determine which rows to pick from lookuptable (each row of rows_wanted correspond to the indicies wanted for that minibatch)
	local rows_wanted = torch.zeros(minibatch_size, K+1):long()
	rows_wanted:narrow(2, 2, K):add(sample_indices:view(1, K):expand(minibatch_size,K))
	rows_wanted:select(2,1):add(output_minibatch)
	--rows_wanted[1] = output_minibatch[1]
	--print(tanh_result)
	local lookuptable_rows = lookuptable:forward(rows_wanted)
	--print(lookuptable_rows)
	--assert(false)
	
	-- Set whether each of these are true word or sampled or not (all are sampled, except the first is true)
	local classifications = torch.zeros(minibatch_size, K+1)
	classifications:select(2,1):fill(1)
	--print(classifications)
	--assert(false)

	-- Get the p_ML for each of these words
	pmls = torch.zeros(minibatch_size, K+1)
	pmls:select(2,1):add(probs:index(1, output_minibatch))
	pmls:narrow(2, 2, K):add(probs:index(1, sample_indices):view(1, K):expand(minibatch_size,K))
	--print(pmls)

	--pmls[1] = probs[output_minibatch[1]]
	--pmls:narrow(1, 2, K):add(probs:index(1, sample_indices))
	local z = torch.zeros(minibatch_size, K+1)
	for i = 1, minibatch_size do
		--print(tanh_result[i])
		z[i] = torch.mm(tanh_result[i]:view(1, tanh_result:size(2)), lookuptable_rows[i]:t())
	end
	--print(z)

	--local z = torch.mm(tanh_result, lookuptable_rows)


	local subtracted = z - torch.log(torch.mul(pmls, K))
	--print(subtracted)

	--assert(false)

	local sigmoid = nn.Sigmoid()
	-- Calculate the prediction for each of the inputs
	local predictions = sigmoid:forward(subtracted)
	--print(predictions)
	criterion = nn.BCECriterion()

	-- Calculate the loss
	local loss = criterion:forward(predictions, classifications)
	--print(loss)

	-- Calculate the gradient
	local dLdpreds = criterion:backward(predictions, classifications) -- gradients of loss wrt preds
	local sigmoid_grad = sigmoid:backward(subtracted, dLdpreds)
	--print(tanh_result)
	--print(sigmoid_grad)

	local lookup_grad = torch.zeros(lookuptable_rows:size())
	--print(lookup_grad)
	for i = 1, minibatch_size do
		--print(tanh_result[i]:size(), sigmoid_grad[i]:size())
		lookup_grad[i] = torch.mm(sigmoid_grad[i]:view(K+1, 1), tanh_result[i]:view(1, tanh_result:size(2)))
		--print(tanh_result[i])
		--z[i] = torch.mm(tanh_result[i]:view(1, tanh_result:size(2)), lookuptable_rows[i]:t())
	end
	--print(lookup_grad)


	--print(tanh_result)
	--print(sigmoid_grad)
	--print(lookuptable_rows)

	local model_grad = torch.zeros(tanh_result:size())
	--print(lookup_grad)
	for i = 1, minibatch_size do
		--print(sigmoid_grad[i], lookuptable_rows[i])
		model_grad[i] = torch.mm(sigmoid_grad[i]:view(1, K+1), lookuptable_rows[i])
		--print(tanh_result[i])
		--z[i] = torch.mm(tanh_result[i]:view(1, tanh_result:size(2)), lookuptable_rows[i]:t())
	end
	--print(model_grad)
	--assert(false)


	--local lookup_grad = torch.mm(tanh_result:t(), sigmoid_grad):t()
	--local model_grad = torch.mm(sigmoid_grad, lookuptable_rows)
	-- print(z)
	-- print(tanh_result)
	-- print(lookuptable_rows)
	-- print(sigmoid_grad)
	-- print(lookup_grad)
	-- print(model_grad)

	lookuptable:backward(rows_wanted, lookup_grad)
	lookuptableparameters:add(torch.mul(lookuptablegradparameters,-1*eta))

	model:backward(input_minibatch, model_grad)
	modelparams:add(torch.mul(modelgradparams,-1*eta))

	--local lookup_grad = torch.cmul(z, sigmoid_grad:expand(minibatch_size, hidden_size))


end

	
	-- local minibatch_size, hidden_size = z:size(1), z:size(2)
	-- --print(lookuptable.weight)
	-- local total_gradient = torch.zeros(z:size())


	-- lookuptablegradparameters:zero()

	-- -- Grab the ML probabililities for each of the true words in the minibatch
	-- local prob = probs:index(1, output_minibatch)

	-- -- Grab the lookuptable rows for each of the true words. This is minibatchsize x hiddenlayer
	-- local lookuptable_rows = lookuptable:forward(output_minibatch)

	-- -- We now dot product the ith row of z with the ith row of lookuptable_rows (elementwise-mul and then sum left to right)
	-- -- The product will be minibatchsize x 1
	-- local dot_product = torch.cmul(z, lookuptable_rows):sum(2)
	-- local subtracted = dot_product - torch.log(torch.mul(prob, K))


	-- local sigmoid = nn.Sigmoid()
	-- -- Calculate the prediction for each of the inputs
	-- local predictions = sigmoid:forward(subtracted)
	-- criterion = nn.BCECriterion()
	-- -- Calculate the loss - note that these are all the correct ones, so the correct classes are all just 1
	-- local loss = criterion:forward(predictions, torch.ones(predictions:size(1)))
	-- --print(loss)

	-- -- Calculate the gradient
	-- dLdpreds = criterion:backward(predictions, torch.zeros(predictions:size(1))) -- gradients of loss wrt preds
	-- sigmoid:backward(subtracted, dLdpreds)
	
	-- local resizedGrad = (sigmoid.gradInput):view(minibatch_size, 1)
	-- -- Update the lookuptable 
	-- local lookup_grad = torch.cmul(z, resizedGrad:expand(minibatch_size, hidden_size))

	-- local overall_grad = torch.cmul(lookuptable_rows, resizedGrad:expand(minibatch_size, hidden_size))
	-- -- Add that to the gradient to be passed back to the model
	-- total_gradient:add(lookup_grad)
	-- --print(lookup_grad)
	-- lookuptable:backward(output_minibatch, lookup_grad)
	-- lookuptableparameters:add(torch.mul(lookuptablegradparameters,-1*eta))


	-- -- do the sampled cases
	-- for sample_i =1, sample_indices:size(1) do
	-- 	lookuptablegradparameters:zero()
	-- 	local idx = sample_indices[sample_i]
	-- 	local prob = probs[idx]

	-- 	local lookuptable_row = lookuptable:forward(torch.LongTensor{idx}):squeeze()

	-- 	-- minibatchsize x 1
	-- 	local dot_product = torch.mv(z, lookuptable_row)
	-- 	local subtracted = dot_product - torch.log(K*prob)
	-- 	local sigmoid = nn.Sigmoid()
	-- 	local predictions = sigmoid:forward(subtracted)

	-- 	criterion = nn.BCECriterion()
	-- 	local loss = criterion:forward(predictions, torch.zeros(predictions:size(1)))

	-- 	dLdpreds = criterion:backward(predictions, torch.zeros(predictions:size(1))) -- gradients of loss wrt preds
	-- 	sigmoid:backward(subtracted, dLdpreds)

	-- 	local resizedGrad = (sigmoid.gradInput):view(minibatch_size, 1)
	-- 	local lookup_grad = torch.cmul(z, resizedGrad:expand(minibatch_size, hidden_size))

	-- 	--lookuptable_row:view(1, lookuptable_row:size(1)):expand(minibatch_size, hidden_size)

	-- 	local overall_grad = torch.cmul(lookuptable_row:view(1, lookuptable_row:size(1)):expand(minibatch_size, hidden_size), resizedGrad:expand(minibatch_size, hidden_size))

	-- 	total_gradient:add(lookup_grad)
	-- 	--print(lookup_grad)
	-- 	lookuptable:backward(torch.LongTensor{idx}, lookup_grad:sum(1))
	-- 	lookuptableparameters:add(torch.mul(lookuptablegradparameters,-1*eta))
	-- end

	-- model:backward(input_minibatch, total_gradient)
	-- modelparams:add(torch.mul(modelgradparams,-1*eta))

--end
