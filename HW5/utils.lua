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

-- sentences_sparse, sentences_dense, and outputs are tables of 2D tensors, where the ith element of the table
-- is a 2D tensor associated with the ith sentence.
-- Return: Table where the ith element is a 1D tensor with the class predictions for the ith sentence.
function predict_each_sentence(sentences_sparse, sentences_dense, outputs, nclasses, predictor, start_class)

	local predicted_outputs = {}
	for i = 1, #sentences_sparse do
		predicted_outputs[i] = viterbi(sentences_sparse[i], predictor, nclasses, start_class, sentences_dense[i])
	end
	return predicted_outputs
end

function find_kaggle_dims(output, start_class, end_class, o_class)

	max_span = 0
	max_classes = 0
	n = output:size(1)
	sentences = 0

	-- one pass through to get max classes and indexes
	local previous_class = o_class
	local this_span_length = 0
	local this_class_length = 0

	for i=1, n do
		local this_class = output[i]
		-- finish last sentence
		if (this_class == start_class) then
			sentences = sentences + 1
			if this_class_length > max_classes then
				max_classes = this_class_length
			end
			this_class_length = 0
		-- finish last span
		elseif (this_class == o_class) then
			if this_span_length > max_span then
				max_span = this_span_length
			end
			this_span_length = 0
		-- some nontrivial class
		elseif (this_class ~= end_class) then
			-- ongoing span
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

	return max_span, max_classes, sentences
end

-- generate sentences x max_classes x max_span+1 tensor for kaggle
-- the first entry will be the id of the class
-- [i][j][k] = the kth index in the span of the jth named entity of the ith sentence
-- zeros are padding
function kagglify_output(output, start_class, end_class, o_class, max_span, max_classes, sentences)

	kaggle_output = torch.zeros(sentences, max_classes, max_span+1)

	previous_class = o_class
	local this_sentence_idx = 0
	local this_class_idx = 0
	local this_span_idx = 2

	for i=1, n do
		local this_class = output[i]
		-- finish last sentence
		if (this_class == start_class) then
			this_sentence_idx = this_sentence_idx + 1
			this_class_idx = 0
			this_span_idx = 2
		-- finish last span
		elseif (this_class == o_class) then
			this_span_idx = 2
		-- some nontrivial class
		elseif (this_class ~= end_class) then
			-- ongoing span
			if (this_class == previous_class) then
				kaggle_output[this_sentence_idx][this_class_idx][this_span_idx] = i
				this_span_idx = this_span_idx + 1
			-- new span
			else
				this_class_idx = this_class_idx + 1
				if (this_class_idx > max_classes) then
					print(this_class_idx, max_classes, i, this_sentence_idx)
				end
				kaggle_output[this_sentence_idx][this_class_idx][1] = this_class
				kaggle_output[this_sentence_idx][this_class_idx][2] = i
				this_span_idx = 3
			end
		end
		previous_class = this_class
	end

	return kaggle_output, max_span, max_classes, sentences

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

