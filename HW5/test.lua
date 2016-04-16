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


function beam_viterbi_efficiency()
	n = 100
	k = 10
	beam_search_alg = wrap_beam_search(k)
	timer = torch.Timer()
	start_class = 1
	for c = 5, 100, 5 do
		r = torch.rand(c, n, c)
		local predictor = (function (ci1, x) return r[ci1][x] end)
		local x = torch.ceil(torch.rand(n-2):add(0.00001):mul(n))
		-- start viterbi
		timer:reset()
		viterbi(x, predictor, c, start_class)
		viterbi_time = timer:time().real
		-- start beam search
		timer:reset()
		beam_search_alg(x, predictor, c, start_class)
		beam_time = timer:time().real
		print(c, viterbi_time, beam_time)
	end
end

function beam_viterbi_accuracy()
	n = 100
	c = 30
	r = torch.rand(c, n, c)
	start_class = 1
	predictor = (function (ci1, x) return r[ci1][x] end)
	x = torch.ceil(torch.rand(n-2):add(0.00001):mul(n))

	for k=1, 25 do
		correct_seq = viterbi(x, predictor, c, start_class)
		beam_seq = beam_search(k, x, predictor, c, start_class)
		accuracy = torch.mean(torch.eq(correct_seq, beam_seq):double())
		print(k, accuracy)
	end
end




--test_viterbi(wrap_beam_search(300))
--test_viterbi(viterbi)
--test_split_data_into_sentences()

--beam_viterbi_efficiency()
beam_viterbi_accuracy()