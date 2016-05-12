require 'nn'
require 'rnn'
require 'optim'



function memm_model(nfeatures, nclasses, embeddingsize, hidden)
	local parallel_table = nn.ParallelTable()
	local prev_class_part = nn.Sequential()
	prev_class_part:add(nn.LookupTable(nclasses, embeddingsize))
	prev_class_part:add(nn.View(-1):setNumInputDims(2))

	prev_class_part:add(nn.Linear(embeddingsize, hidden))
	local input_part = nn.Linear(nfeatures, hidden)
	
	parallel_table:add(input_part)
	parallel_table:add(prev_class_part)

	local model = nn.Sequential()
	model:add(parallel_table)
	model:add(nn.CAddTable())

	model:add(nn.Linear(hidden, nclasses))

	model:add(nn.LogSoftMax())

	criterion = nn.ClassNLLCriterion()

	if usecuda then
		model:cuda()
		print("Converted model to CUDA")
		criterion:cuda()
		print("Converted crit to CUDA")
	end

	return model,criterion
end

-- This expects inputs to NOT BE transposed. For example, if you have b sequences of length l, where at each step you are looking 
-- at a window of size dwin, the dimensions of what should be passed into this model are b x l x dwin.
function bidirectionalRNNmodel(vocab_size, embed_dim, output_dim, rnn_unit1, rnn_unit2, dropout, usecuda, hidden, num_layers, dwin)
	batchLSTM = nn.Sequential()

	-- This is needed to deal with SplitTable being stupid about LongTensors
	local copy = nn.Copy('torch.LongTensor', 'torch.DoubleTensor')
	local firstsplit = nn.SplitTable(1,3)
	-- This is needed to deal with LookupTable not having updateGradOutput
	copy.updateGradInput = function() end
	firstsplit.updateGradInput = function() end

	batchLSTM:add(copy)
	batchLSTM:add(firstsplit)
	local embedding = nn.Sequencer(nn.LookupTable(vocab_size, embed_dim))
	batchLSTM:add(embedding) --will return a sequence-length x batch-size x embedDim tensor
	batchLSTM:add(nn.Sequencer(nn.View(-1):setNumInputDims(2)))
	batchLSTM:add(nn.Sequencer(nn.Unsqueeze(2)))
	batchLSTM:add(nn.JoinTable(1, 2))

	batchLSTM:add(nn.SplitTable(1, 3)) --splits into a sequence-length table with batch-size x embedDim entries
	

	local sequencers = {}

	biseq = nil
	for layer = 1, num_layers do 
		if layer == 1 then 
			biseq = nn.BiSequencer(nn.FastLSTM(dwin*embed_dim, dwin*embed_dim), nn.FastLSTM(dwin*embed_dim, dwin*embed_dim))
		else
			biseq = nn.BiSequencer(nn.FastLSTM(2*dwin*embed_dim, dwin*embed_dim), nn.FastLSTM(2*dwin*embed_dim, dwin*embed_dim))
		end
		sequencers[layer] = biseq
		batchLSTM:add(biseq)
		batchLSTM:add(nn.Sequencer(nn.Dropout(dropout)))
	end
	batchLSTM:add(nn.Sequencer(nn.Linear(2*dwin*embed_dim, hidden)))
	batchLSTM:add(nn.Sequencer(nn.ReLU()))
	batchLSTM:add(nn.Sequencer(nn.Dropout(dropout)))
	batchLSTM:add(nn.Sequencer(nn.Linear(hidden, output_dim)))
	batchLSTM:add(nn.Sequencer(nn.LogSoftMax()))

	crit = nn.SequencerCriterion(nn.ClassNLLCriterion())
	if usecuda then
		batchLSTM:cuda()
		print("Converted LSTM to CUDA")
		crit:cuda()
		print("Converted crit to CUDA")
	end
	print(batchLSTM)
	return batchLSTM, crit, sequencers
end


-- This expects inputs to NOT BE transposed. 
-- The input's should be batched though. In particular, the input to this model should be
-- b x n x w, where b is the number of batches, n is the sequence in the batch, and w is the # of features for each seq element.
function bidirectionalRNNmodelExtraFeatures(num_features, embed_dim, output_dim, rnn_unit1, rnn_unit2, dropout, usecuda, hidden, num_layers)
	batchLSTM = nn.Sequential()
	batchLSTM:add(nn.Transpose{1,2})
	batchLSTM:add(nn.SplitTable(1,3))

	batchLSTM:add(nn.Sequencer(nn.Linear(num_features, embed_dim)))


	local sequencers = {}

	biseq = nil
	for layer = 1, num_layers do 
		if layer == 1 then 
			biseq = nn.BiSequencer(nn.FastLSTM(embed_dim, embed_dim), nn.FastLSTM(embed_dim, embed_dim))
		else
			biseq = nn.BiSequencer(nn.FastLSTM(2*embed_dim, embed_dim), nn.FastLSTM(2*embed_dim, embed_dim))
		end
		sequencers[layer] = biseq
		batchLSTM:add(biseq)
		batchLSTM:add(nn.Sequencer(nn.Dropout(dropout)))
	end
	batchLSTM:add(nn.Sequencer(nn.Linear(2*embed_dim, hidden)))
	batchLSTM:add(nn.Sequencer(nn.ReLU()))
	batchLSTM:add(nn.Sequencer(nn.Dropout(dropout)))
	batchLSTM:add(nn.Sequencer(nn.Linear(hidden, output_dim)))
	batchLSTM:add(nn.Sequencer(nn.LogSoftMax()))

	crit = nn.SequencerCriterion(nn.ClassNLLCriterion())
	if usecuda then
		batchLSTM:cuda()
		print("Converted LSTM to CUDA")
		crit:cuda()
		print("Converted crit to CUDA")
	end
	print(batchLSTM)
	return batchLSTM, crit, sequencers
end

-- This expects inputs to NOT BE transposed. 
-- The input's should be batched though. In particular, the input to this model should be
-- b x n x w, where b is the number of batches, n is the sequence in the batch, and w is the # of features for each seq element.
function biLSTMMEMM(num_features, embed_dim, output_dim, rnn_unit1, rnn_unit2, dropout, usecuda, hidden, num_layers)

	parallel_table = nn.ParallelTable()

	prev_class_part = nn.Sequential()
	-- This is needed to deal with SplitTable being stupid about LongTensors
	local copy = nn.Copy('torch.LongTensor', 'torch.DoubleTensor')
	local firstsplit = nn.SplitTable(2,3)
	-- This is needed to deal with LookupTable not having updateGradOutput
	copy.updateGradInput = function() end
	firstsplit.updateGradInput = function() end

	prev_class_part:add(copy)
	prev_class_part:add(firstsplit)
	prev_class_part:add(nn.Sequencer(nn.LookupTable(output_dim, output_dim)))
	prev_class_part:add(nn.Sequencer(nn.View(-1):setNumInputDims(2)))
	prev_class_part:add(nn.Sequencer(nn.Unsqueeze(1)))
	prev_class_part:add(nn.JoinTable(1, 3))

	batchLSTM = nn.Sequential()
	batchLSTM:add(nn.SplitTable(2,3))
	batchLSTM:add(nn.Sequencer(nn.Linear(num_features, embed_dim)))

	local sequencers = {}

	biseq = nil
	for layer = 1, num_layers do 
		if layer == 1 then 
			biseq = nn.BiSequencer(nn.FastLSTM(embed_dim, embed_dim), nn.FastLSTM(embed_dim, embed_dim))
		else
			biseq = nn.BiSequencer(nn.FastLSTM(2*embed_dim, embed_dim), nn.FastLSTM(2*embed_dim, embed_dim))
		end
		sequencers[layer] = biseq
		batchLSTM:add(biseq)
		batchLSTM:add(nn.Sequencer(nn.Dropout(dropout)))
	end
	batchLSTM:add(nn.Sequencer(nn.Linear(2*embed_dim, hidden)))

	-- convert back to tensor so we can concat
	batchLSTM:add(nn.Sequencer(nn.Unsqueeze(1)))
	batchLSTM:add(nn.JoinTable(1, 3))

	parallel_table:add(batchLSTM)
	parallel_table:add(prev_class_part)
	lstmMEMM = nn.Sequential()
	lstmMEMM:add(parallel_table)
	lstmMEMM:add(nn.JoinTable(3, 3))

	--split back before linear and softmax
	lstmMEMM:add(nn.SplitTable(1, 3))

	-- add a linear and a softmax
	output_layer = nn.Sequential()
	output_layer:add(nn.Linear(hidden+output_dim, output_dim))
	output_layer:add(nn:LogSoftMax())
	lstmMEMM:add(nn.Sequencer(output_layer))

	crit = nn.SequencerCriterion(nn.ClassNLLCriterion())
	if usecuda then
		lstmMEMM:cuda()
		print("Converted LSTM to CUDA")
		crit:cuda()
		print("Converted crit to CUDA")
	end
	print(lstmMEMM)
	return lstmMEMM, batchLSTM, output_layer, prev_class_part, crit, sequencers
end

function rnn_model(vocab_size, embed_dim, output_dim, rnn_unit1, rnn_unit2, dropout, usecuda) 
	batchLSTM = nn.Sequential()
	local embedding = nn.LookupTable(vocab_size, embed_dim)
	batchLSTM:add(embedding) --will return a sequence-length x batch-size x embedDim tensor
	batchLSTM:add(nn.Transpose({1,2}))
	batchLSTM:add(nn.SplitTable(1, 3)) --splits into a sequence-length table with batch-size x embedDim entries

	-- Add the first layer rnn unit
	if rnn_unit1 == 'lstm' then
		batchLSTM:add(nn.Sequencer(nn.FastLSTM(embed_dim, embed_dim)))
		print("Unit1: LSTM added")
	elseif rnn_unit1 == 'gru' then 
		batchLSTM:add(nn.Sequencer(nn.GRU(embed_dim, embed_dim)))
		print("Unit1: GRU added")
	else
		print("Invalid unit 1")
		assert(false)
	end

	-- If there is a second layer, add dropout and the layer.
	if rnn_unit2 ~= 'none' then
		batchLSTM:add(nn.Sequencer(nn.Dropout(dropout)))
		print("Dropout added", dropout)
		-- Add second layer 
		if rnn_unit2 == 'lstm' then
			batchLSTM:add(nn.Sequencer(nn.FastLSTM(embed_dim, embed_dim)))
			print("Unit2: LSTM added")
		elseif rnn_unit2 == 'gru' then 
			batchLSTM:add(nn.Sequencer(nn.GRU(embed_dim, embed_dim)))
			print("Unit2: GRU added")
		else
			print("Invalid unit 2")
			assert(false)
		end
	else
		print("No unit 2")
	end
	batchLSTM:add(nn.Sequencer(nn.Linear(embed_dim, output_dim)))
	batchLSTM:add(nn.Sequencer(nn.LogSoftMax()))

	-- Add a criterion
	crit = nn.SequencerCriterion(nn.ClassNLLCriterion())
	if usecuda then
		batchLSTM:cuda()
		print("Converted LSTM to CUDA")
		crit:cuda()
		print("Converted crit to CUDA")
	end
	return batchLSTM, crit, embedding

end
