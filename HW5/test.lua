dofile("utils.lua")




function make_predictor(model)
	return (function(a, b) return model + a + b end)
end

x = torch.Tensor{1,2,3}

-- predictor = make_predictor(x)
-- x:add(1)
-- print(predictor(1, 2))


-- expect 
--  3
--  1
--  1
--  1
--  2
--  2
--  2
--  2
--  2
--  2
-- [torch.DoubleTensor of size 10]
function test_viterbi(viterbi_alg)
	local x = torch.Tensor{5, 3, 3, 2, 1, 2, 4, 3, 1, 1}:resize(10, 1)
	local predictor = (function (ci1, x) 
						if (x[1] == 5) then
							return torch.Tensor{0, 0, 1.0}
						elseif (ci1 == 3) then
							return torch.Tensor{0.5, 0.5, 0}
						elseif (ci1 == 1) then
							if ((x[1] == 1) or (x[1] == 4)) then
								return torch.Tensor{0.1, 0.15, 0}
							else
								return torch.Tensor{0.15, 0.1, 0}
							end
						else
							if ((x[1] == 1) or (x[1] == 4)) then
								return torch.Tensor{0.08, 0.18, 0}
							else
								return torch.Tensor{0.12, 0.12, 0}
							end
						end
						end)

	local numClasses = 3
	local start_class = 3
	local yhat = viterbi_alg(x, predictor, numClasses, start_class)
	print(yhat)
end


function test_split_data_into_sentences()

	sparse_inputs = torch.randn(10, 6):long()
	dense_inputs = torch.randn(10, 6)
	output_classes = torch.LongTensor{8, 1, 2, 3, 9, 8, 3, 2, 1, 9}
	end_class = 9
	st, dt, ot = split_data_into_sentences(sparse_inputs, dense_inputs, output_classes, end_class)

	print(ot[2])

end

-- test_viterbi(viterbi)
test_split_data_into_sentences()
