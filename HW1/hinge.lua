function hingeLoss(W, b, Xs, Ys, lambda)
	local N = Xs:size()[1]
	local K = W:size()[1]
	local softmax_res = softmax(Xs, W, b)
	local total = 0.0
	for n = 1, N do
		--true class probability
		local yc = 0
		--greatest false class probability
		local ycp = 0
		for k = 1, K do
			if (Ys[n] == k) then
				yc = softmax_res[n][k]
			elseif (Ys[n] > ycp) then
				ycp = Ys[n]
			end
		end
		total = total+math.max(0, 1-(yc + ycp))
	end
	return ((-1)*total) + .5*lambda*torch.pow(W,2):sum()
end

-- Calculates the gradient of W
--      Will be using stochastic gradient descent with minibatches
-- Inputs:
--       W:           weights (nclasses x nfeatures)
--       Xs:          input features (sparse representation)
-- 		 Ys:          output classes (N x 1)
--       start_index: index of start of minibatch
--       end_index:   index of end of minibach
function hingeGradient(W, b, Xs, Ys, start_index, end_index)
	TODO
end


