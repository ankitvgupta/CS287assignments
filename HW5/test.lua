dofile("utils.lua")




function make_predictor(model)
	return (function(a, b) return model + a + b end)
end

x = torch.Tensor{1,2,3}

predictor = make_predictor(x)
x:add(1)
print(predictor(1, 2))


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

--test_viterbi(viterbi)
