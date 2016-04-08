-- x is a sequence of inputs
-- predictor is a function that takes in past class and xi and
--   provides a
function viterbi(x, predictor, numClasses, start_class, debugger)
	local n = x:size(1)
	local pi = torch.ones(n, numClasses):mul(-1e+31)
	local bp = torch.ones(n, numClasses)
	pi[1][start_class] = 0

	for i=2, n do
		for ci1=1, numClasses do
			local yci1 = predictor(ci1, x[i])
			for ci=1, numClasses do
				local v = pi[i-1][ci1] + torch.log(yci1[ci])
				if v > pi[i][ci] then
					pi[i][ci] = v
					bp[i][ci] = ci1
				end
			end
		end
	end

	local yhat = torch.Tensor(n)

	local lastBestClass = 1
	local lastBestScore = pi[n][1]
	for ci=2, numClasses do
		local score = pi[n][ci]
		if score > lastBestScore then
			lastBestScore = score
			lastBestClass = ci
		end
	end

	yhat[n] = lastBestClass

	for i=n-1, 1,-1 do
		yhat[i] = bp[i+1][yhat[i+1]]
	end

	if debugger ~= nil then
		print("bp:", bp)
		print("pi:", pi)
	end

	return yhat
end

-- expected output:
-- pi:	 0.0000e+00 -1.0000e+31 -1.0000e+31
--  2.0000e+00  1.0000e+00  0.0000e+00
--  5.0000e+00  4.0000e+00  3.0000e+00
--  1.1000e+01  1.0000e+01  9.0000e+00
-- [torch.DoubleTensor of size 4x3]

--  1
--  3
--  3
--  1
function test_viterbi(viterbi_alg)
	local x = torch.Tensor{1, 3, 2, 3}:resize(4, 1)
	local predictor = function (ci1, x) 
						return torch.Tensor{torch.exp(ci1*x[1]-1),
											torch.exp(ci1*x[1]-2),
											torch.exp(ci1*x[1]-3)} end
	local numClasses = 3
	local start_class = 1
	local yhat = viterbi_alg(x, predictor, numClasses, start_class, true)
	print(yhat)
end

test_viterbi(viterbi)
