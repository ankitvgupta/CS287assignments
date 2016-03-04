dofile(_G.path.."utils.lua")


-- Returns F and N. F is the counts matrix, and N is the indicator matrix which denotes whether each feature_class pair is greater than 1.
--function createCountsMatrix(sparse_training_input, dense_training_input, training_output, n_sparse_features, nclasses)
function createCountsMatrix(sparse_training_input, training_output, n_sparse_features, nclasses)
	--print("Num items", sparse_training_input:size(1), dense_training_input:size(1))
	--local num_dense_features = dense_training_input:size(2)
	--local F = torch.zeros(n_sparse_features + dense_training_input:size(2), nclasses):long()
	local F = torch.zeros(n_sparse_features, nclasses):long()
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
	--local last_rows = F:narrow(1, F:size()[1]-num_dense_features+1, num_dense_features)

	--for class_val = 1,nclasses do
		-- Gets the indices of the rows that are in this class
	--	local indices_for_this_class = torch.range(1,training_output:size(1)):long()[torch.eq(training_output, class_val)]

		-- Gets the rows at those indicies and sums them
	--	local features_sum = dense_training_input:index(1, indices_for_this_class):sum(1):squeeze()

		-- Adds those to the right class in last_rows
	--	last_rows:select(2, class_val):add(features_sum)
	end

	return F
end


-- Returns W, b. For Naive Bayes,
-- 	W is log(p(x|y))
-- 	b is log(p(y)).
function naiveBayes(sparse_training_input, dense_training_input, training_output, nsparse_features, nclasses, alpha)

	local F = createCountsMatrix(sparse_training_input, dense_training_input, training_output, nsparse_features, nclasses):double()

	-- Add a small offset for smoothing
	F = F + alpha

	nfeatures = nsparse_features + dense_training_input:size(2)

	-- Now, we column normalize the Tensor
	sum_of_each_col = torch.sum(F, 1)
	p_x_given_y = torch.Tensor(nfeatures, nclasses):zero()
	for n = 1, F:size()[1] do
		p_x_given_y[n] = torch.cdiv(F[n] , sum_of_each_col)
	end
	print("Sum", p_x_given_y:sum())

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
	
	return W:t(), b

end

-- Arguments
	-- Context is a tensor containing the input words
	-- W_array is the array of W tensors for each context size
	-- b_array is the array of b tensors for each context size
	-- F_array is the array of F tensors for each context size
	-- N_array is the array of N tensors for each context size
function p_wb(context, W_array, b_array, F_array, N_array)

	-- This tells us how many words constitute the context being used
	local context_size = #context

	-- Grab the tensors associated with this context size
	local W = W_array[context_size]
	local b = b_array[context_size]
	local F = F_array[context_size]
	local N = N_array[context_size]

	local mult = sparseMultiply_nopadding(context, W)
	local z = mult + b:expand(mult:size(1), b:size(2))

	local softmaxlayer = nn.SoftMax()
	local p_ml = softmaxlayer:forward(z)

	





end

function whitten_bell()

	return

end

