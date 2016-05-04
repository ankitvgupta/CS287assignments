require 'nn'
require 'rnn'


batchLSTM = nn.Sequential()
vocab_size = 20
embed_dim = 3
output_dim = 5
dwin = 3
local inputs = (torch.randn(4, 6, dwin):abs():mul(4):long() + 1)
--print(inputs)
batchLSTM:add(nn.Copy('torch.LongTensor', 'torch.DoubleTensor'))
batchLSTM:add(nn.SplitTable(1, 3))
--print(batchLSTM:forward(inputs))
local embedding = nn.Sequencer(nn.LookupTable(vocab_size, embed_dim))
--local embedding = nn.TemporalConvolution(3, embed_dim, 1, 1)
print(batchLSTM:forward(inputs)[1])
batchLSTM:add(embedding) --will return a sequence-length x batch-size x embedDim tensor
print(batchLSTM:forward(inputs)[1])
batchLSTM:add(nn.Sequencer(nn.View(-1):setNumInputDims(2)))
batchLSTM:add(nn.Sequencer(nn.Unsqueeze(2)))

batchLSTM:add(nn.JoinTable(1, 2))
batchLSTM:add(nn.SplitTable(1, 3))
--print(batchLSTM:forward(inputs))


--batchLSTM:add(nn.Sequencer(nn.FastLSTM(embed_dim, embed_dim)))
biseq = nn.BiSequencer(nn.FastLSTM(dwin*embed_dim, dwin*embed_dim), nn.FastLSTM(dwin*embed_dim, dwin*embed_dim), 1)
batchLSTM:add(biseq)
--print(batchLSTM:forward(inputs))
--print("Test")
--print(nn.JoinTable(1,1):forward({torch.randn(4, 30), torch.randn(4, 30)}))
--print(biseq.bwdSeq)

batchLSTM:add(nn.Sequencer(nn.Linear(2*dwin*embed_dim, output_dim)))
batchLSTM:add(nn.Sequencer(nn.LogSoftMax()))
crit = nn.SequencerCriterion(nn.ClassNLLCriterion())
print(batchLSTM)
preds = batchLSTM:forward(inputs)
loss = crit:forward(preds, torch.ones(6, 4))
print(loss)
dLdpreds = crit:backward(preds, torch.ones(6, 4)) -- gradients of loss wrt preds
batchLSTM:backward(inputs, dLdpreds)

--print(batchLSTM:forward(inputs))
--print(batchLSTM:forward(inputs)[1])

