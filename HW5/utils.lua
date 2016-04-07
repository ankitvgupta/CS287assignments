-- x is a sequence of inputs
-- predictor is a function that takes in xi and past class and
--   provides a
-- for testing 
function slow_viterbi(x, predictor, classes, start_class, debugger)
	local n = x:size(1)
	local numClasses = classes:size(1)

	local pi = torch.ones(n+1, numClasses):mul(-1e+31)
	pi[1][start_class] = 0

	for i=2, n do
		for ci=1, numClasses do
			local maxval = -1e+31
			for ci1=1, numClasses do
				local v = pi[i-1][ci1] + torch.log(predictor(ci1, x[i-1])[ci])
				if v > maxval then
					maxval = v
				end
			end
			pi[i][ci] = maxval
		end
	end

	local yhat = torch.Tensor(n)
	for i=1, n do
		local bestClass = 1
		local bestScore = pi[i][1]
		for ci=2, numClasses do
			if pi[i][ci] > bestScore then
				bestClass = ci
				bestScore = pi[i][ci]
			end
		end
		yhat[i] = classes[bestClass]
	end

	if debugger ~= nil then
		print("pi:", pi)
	end

	return yhat
end


-- expected output:
-- pi:	 0.0000e+00 -1.0000e+31 -1.0000e+31
--  0.0000e+00 -1.0000e+00 -2.0000e+00
--  6.0000e+00  5.0000e+00  4.0000e+00
--  9.0000e+00  8.0000e+00  7.0000e+00
-- -1.0000e+31 -1.0000e+31 -1.0000e+31
-- [torch.DoubleTensor of size 5x3]

--  1
--  1
--  1
--  1
function test_viterbi(viterbi_alg)
	local x = torch.Tensor{1, 3, 2, 3}:resize(4, 1)
	local predictor = function (ci1, x) 
						return torch.Tensor{torch.exp(ci1*x[1]-1),
											torch.exp(ci1*x[1]-2),
											torch.exp(ci1*x[1]-3)} end
	local classes = torch.Tensor{1, 2, 3}
	local start_class = 1
	local yhat = viterbi_alg(x, predictor, classes, start_class, true)
	print(yhat)
end


test_viterbi(slow_viterbi)

