require 'nn'
require 'rnn'


batchLSTM = nn.Sequential()
vocab_size = 20
embed_dim = 9
output_dim = 5
dwin = 1
local inputs = (torch.randn(4, 6, dwin):abs():mul(4):long() + 1):transpose(1,2)
print(inputs)

batchLSTM:add(nn.SplitTable(1, 3))

print(batchLSTM:forward(inputs))
local embedding = nn.Sequencer(nn.LookupTable(vocab_size, embed_dim))
--local embedding = nn.TemporalConvolution(3, embed_dim, 1, 1)
batchLSTM:add(embedding) --will return a sequence-length x batch-size x embedDim tensor
batchLSTM:add(nn.Sequencer(nn.View(-1):setNumInputDims(2)))
print(batchLSTM:forward(inputs))
batchLSTM:add(nn.Sequencer(nn.Unsqueeze(2)))
print(batchLSTM:forward(inputs))
batchLSTM:add(nn.JoinTable(1, 2))
print(batchLSTM:forward(inputs))
batchLSTM:add(nn.View(-1, 6, dwin*embed_dim))
--print(batchLSTM:forward(inputs))
batchLSTM:add(nn.Transpose({1,2}))
batchLSTM:add(nn.SplitTable(1, 3))
print(batchLSTM:forward(inputs))

--batchLSTM:add(nn.Sequencer(nn.FastLSTM(embed_dim, embed_dim)))
biseq = nn.BiSequencer(nn.FastLSTM(dwin*embed_dim, dwin*embed_dim), nn.FastLSTM(dwin*embed_dim, dwin*embed_dim), 1)
batchLSTM:add(biseq)
--print(batchLSTM:forward(inputs))
--print("Test")
--print(nn.JoinTable(1,1):forward({torch.randn(4, 30), torch.randn(4, 30)}))
--print(biseq.bwdSeq)

batchLSTM:add(nn.Sequencer(nn.Linear(2*dwin*embed_dim, output_dim)))
batchLSTM:add(nn.Sequencer(nn.LogSoftMax()))
--print(batchLSTM:forward(inputs))
--print(batchLSTM:forward(inputs)[1])

