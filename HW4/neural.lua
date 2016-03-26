require 'nn'
require 'rnn'
function rnn(vocab_size, embed_dim, output_dim)
	batchLSTM = nn.Sequential()
	batchLSTM:add(nn.LookupTable(vocab_size, embed_dim)) --will return a sequence-length x batch-size x embedDim tensor

	-- 1 indicates the dimension we are splitting along. 3 indicates the number of dimensions in the input (allows for batching)
	batchLSTM:add(nn.SplitTable(1, 3)) --splits into a sequence-length table with batch-size x embedDim entries
	-- now let's add the LSTM stuff
	batchLSTM:add(nn.Sequencer(nn.LSTM(embed_dim, embed_dim)))
	batchLSTM:add(nn.Sequencer(nn.Linear(embed_dim, output_dim)))
	batchLSTM:add(nn.Sequencer(nn.LogSoftMax()))

	-- Add a criterion
	crit = nn.SequencerCriterion(nn.ClassNLLCriterion())

	return batchLSTM, crit
end

r, crit = rnn(100, 5, 2)

-- batchsize = 5. SequenceLength = 3. 
inputs = (torch.abs(torch.randn(5,3)*10) + 1):long()
-- batchSize = 5. These outputs will be either 1 or 2.
-- There are 3 of these, since ther eis one for every element in sequence.
batch_output = torch.round(torch.rand(5)) + 1
outputs = {batch_output, batch_output, batch_output}
-- Make some predictions
res = r:forward(inputs:t())
print(res)
loss = crit:forward(res, outputs)
print(loss)
dLdPreds = crit:backward(res, outputs)
print(dLdPreds)
r:backward(inputs:t(), dLdPreds)

