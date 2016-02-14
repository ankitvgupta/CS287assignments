dofile("utils.lua")

-- For odyssey
--dofile("/n/home09/ankitgupta/CS287/CS287assignments/HW2/utils.lua")

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
