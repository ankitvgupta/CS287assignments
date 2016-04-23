function reshape_inputs(num_sequences, input, output)
	assert(input:size(1) == output:size(1))
	local len_wanted = input:size(1) - (input:size(1) % num_sequences)
	local new_inputs = input:narrow(1, 1, len_wanted):reshape(num_sequences, len_wanted/num_sequences)
	local new_outputs = output:narrow(1, 1, len_wanted):reshape(num_sequences, len_wanted/num_sequences)
	return new_inputs, new_outputs
end

function printoptions(opt)
	print("datafile:", opt.datafile, 
		"classifier:", opt.classifier, 
		"alpha:", opt.alpha, 
		"beta:", opt.beta,
		"embedding_size:", opt.embedding_size, 
		"minibatch_size", opt.minibatch_size,
		"optimizer:", opt.optimizer, 
		"epochs:", opt.epochs, 
		"hidden", opt.hidden, 
		"eta:", opt.eta)
end




-- x is a sequence of inputs
-- predictor is a function that takes in past class and xi and
--   provides a
function viterbi(x, predictor, numClasses, start_class, x_dense)
	local n = x:size(1)
	local pi = torch.ones(n, numClasses):mul(-1e+31)
	local bp = torch.ones(n, numClasses)
	local hasDense = (x_dense ~= nil)
	pi[1][start_class] = 0

	for i=2, n do
		for ci1=1, numClasses do
			if (hasDense) then
				yci1 = predictor(ci1, x[i], x_dense[i])
			else
				yci1 = predictor(ci1, x[i])
			end
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

	return yhat
end

function prediction_accuracy(preds, true_vals)
	local total_matches = torch.eq(preds, true_vals):sum()
	return total_matches/preds:size(1)
end
