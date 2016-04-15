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

-- returns kx2 tensor of indices of k max values in t
function argmax_2d(t, K)
	local rows = t:size(1)
	local cols = t:size(2)
	local flatt = t:view(rows*cols)
	local _, sorted_idxs = torch.sort(flatt, true)
	local max_idxs = torch.Tensor(K, 2)
	for i=1, K do
		local this_idx = sorted_idxs[i]
		local this_col_idx = ((this_idx-1) % cols) + 1
		local this_row_idx = ((this_idx-this_col_idx)/cols) + 1
		max_idxs[i][1] = this_row_idx
		max_idxs[i][2] = this_col_idx
	end
	return max_idxs
end

function wrap_beam_search(K)
	return function(x, predictor, numClasses, start_class, x_dense) return beam_search(K, x, predictor, numClasses, start_class, x_dense) end
end



function beam_search(K, x, predictor, numClasses, start_class, x_dense)
	local hasDense = (x_dense ~= nil)
	local n = x:size(1)
	local sequences = torch.Tensor(K, 1)
	local scores = torch.zeros(K)

	for k=1, K do
		sequences[k][1] = start_class
	end

	local hypotheses = torch.Tensor(K, numClasses)
	for i=2, n do
		for k=1, K do
			for ci=1, numClasses do
				if (hasDense) then
					yci1 = predictor(sequences[k][i-1], x[i], x_dense[i])
				else
					yci1 = predictor(sequences[k][i-1], x[i])
				end
				hypotheses[k][ci] = scores[k] + torch.log(yci1[ci])
			end
		end

		max_indices = argmax_2d(hypotheses, K)
		local new_sequences = torch.Tensor(K, i)
		for j=1, K do
			k = max_indices[j][1]
			ci = max_indices[j][2]
			for l=1, i-1 do
				new_sequences[j][l] = sequences[k][l]
			end
			new_sequences[j][i] = ci
		end
		sequences = new_sequences
	end

	return sequences[1]

end

-- sentences_sparse, sentences_dense, and outputs are tables of 2D tensors, where the ith element of the table
-- is a 2D tensor associated with the ith sentence.
-- Return: Table where the ith element is a 1D tensor with the class predictions for the ith sentence.
function predict_each_sentence(viterbi_alg, sentences_sparse, sentences_dense, nclasses, predictor, start_class, includeDense)

	local predicted_outputs = {}
	for i = 1, #sentences_sparse do
		if includeDense then
			predicted_outputs[i] = viterbi_alg(sentences_sparse[i], predictor, nclasses, start_class, sentences_dense[i])
		else
			predicted_outputs[i] = viterbi_alg(sentences_sparse[i]:squeeze(), predictor, nclasses, start_class)
		end
	end
	return predicted_outputs
end

function find_kaggle_dims(output, o_class)

	max_span = 0
	max_classes = 0
	sentences = #output

	-- one pass through to get max classes and indexes
	local previous_class = o_class
	local this_span_length = 0
	local this_class_length = 0

	for s=1, sentences do
		local previous_class = o_class
		local this_span_length = 0
		local this_class_length = 0
		local this_sentence = output[s]
		-- start after <t>, end before </t>
		for c=2, this_sentence:size(1)-1 do
			local this_class = this_sentence[c]
			-- finish last span
			if this_class == o_class then
				if this_span_length > max_span then
					max_span = this_span_length
				end
				this_span_length = 0
			-- part of a span
			else
				-- continuing old span
				if (this_class == previous_class) then
					this_span_length = this_span_length + 1
				-- new span
				else
					this_class_length = this_class_length + 1
					this_span_length = 1
				end
			end
			previous_class = this_class
		end
		-- done with sentence
		if this_class_length > max_classes then
			max_classes = this_class_length
		end
	end
	return max_span, max_classes, sentences
end

-- generate sentences x max_classes x max_span+1 tensor for kaggle
-- the first entry will be the id of the class
-- [i][j][k] = the kth index in the span of the jth named entity of the ith sentence
-- zeros are padding
function kagglify_output(output, o_class, max_span, max_classes, sentences)

	kaggle_output = torch.zeros(sentences, max_classes, max_span+1)

	assert(#output == sentences)

	for s=1, sentences do
		-- +1 for <t>
		word_idx = 0
		previous_class = o_class
		local this_class_idx = 0
		local this_sentence = output[s]
		-- start after <t>, end before </t>
		for c=2, this_sentence:size(1)-1 do
			word_idx = word_idx + 1
			local this_class = this_sentence[c]
			--finish last span
			if (this_class == o_class) then
				this_span_idx = 2
			-- part of a span
			else
				-- continuing old span
				if (this_class == previous_class) then
					kaggle_output[s][this_class_idx][this_span_idx] = word_idx
					this_span_idx = this_span_idx + 1
				-- starting new span
				else
					this_class_idx = this_class_idx + 1
					kaggle_output[s][this_class_idx][1] = this_class
					kaggle_output[s][this_class_idx][2] = word_idx
					this_span_idx = 3
				end
			end
			previous_class = this_class
		end
	end

	return kaggle_output

end


-- [i][j][k] = the kth index in the span of the jth named entity of the ith sentence
-- zeros are padding

function compute_f_score(beta, true_kaggle, pred_kaggle)
	local true_pos = 0 -- number of correct entries in pred_kaggle
	local false_pos = 0 -- number of pred_kaggle entries that are not in true_kaggle
	local false_neg = 0 -- number of entries in true_kaggle not in pred_kaggle

	assert(true_kaggle:size(1) == pred_kaggle:size(1))

	for sentence_idx=1, true_kaggle:size(1) do
		local this_true_sentence = true_kaggle[sentence_idx]
		local this_pred_sentence = pred_kaggle[sentence_idx]
		for true_nem_idx=1, true_kaggle:size(2) do
			local pred_nem_found = false
			local this_true_nem = this_true_sentence[true_nem_idx]
			if (this_true_nem:gt(0):any()) then
				for pred_nem_idx=1, pred_kaggle:size(2) do
					local this_pred_nem = this_pred_sentence[pred_nem_idx]
					if (this_pred_nem:eq(this_true_nem):all()) then
						pred_nem_found = true
						break
					end
				end
				if (pred_nem_found) then
					true_pos = true_pos + 1
				else
					false_neg = false_neg + 1
				end
			end
		end
		for pred_nem_idx=1, pred_kaggle:size(2) do
			local true_nem_found = false
			local this_pred_nem = this_pred_sentence[pred_nem_idx]
			if (this_pred_nem:gt(0):any()) then
				for true_nem_idx=1, true_kaggle:size(2) do
					local this_true_nem = this_true_sentence[true_nem_idx]
					if (this_pred_nem:eq(this_true_nem):all()) then
						true_nem_found = true
						break
					end
				end
				if (not true_nem_found) then
					false_pos = false_pos + 1
				end
			end
		end
	end

	print("true positive:", true_pos, "false positive:", false_pos, "false negative: ", false_neg)

	local fscore = (beta*beta+1)*true_pos/((beta*beta+1)*true_pos+(beta*beta)*false_neg + false_pos)

	return fscore
end

function split_data_into_sentences(sparse_inputs, dense_inputs, output_classes, end_class)
	local sparse_sentence_table = {}
	local dense_sentence_table = {}
	local output_sentence_table = {}
	
	local last_start_index = 1
	local sentence_id = 1

	for i=1, output_classes:size(1) do
		local this_class = output_classes[i]
		-- end the sentence and start a new one
		if (this_class == end_class) then
			local this_sparse_inputs = sparse_inputs:narrow(1, last_start_index, i-last_start_index+1)
			local this_dense_inputs = dense_inputs:narrow(1, last_start_index, i-last_start_index+1)
			local this_output_classes = output_classes:narrow(1, last_start_index, i-last_start_index+1)
			
			sparse_sentence_table[sentence_id] = this_sparse_inputs
			dense_sentence_table[sentence_id] = this_dense_inputs
			output_sentence_table[sentence_id] = this_output_classes

			last_start_index = i+1
			sentence_id = sentence_id + 1
		end
	end
	
	return sparse_sentence_table, dense_sentence_table, output_sentence_table

end

