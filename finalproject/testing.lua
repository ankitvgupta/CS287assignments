require 'nn'
require 'rnn'


batchLSTM = nn.Sequential()
vocab_size = 20
embed_dim = 30
output_dim = 5
local inputs = torch.randn(4, 6):abs():mul(4):long() + 1
print(inputs)

local embedding = nn.LookupTable(vocab_size, embed_dim)
batchLSTM:add(embedding) --will return a sequence-length x batch-size x embedDim tensor
batchLSTM:add(nn.Transpose({1,2}))
batchLSTM:add(nn.SplitTable(1, 3))

--batchLSTM:add(nn.Sequencer(nn.FastLSTM(embed_dim, embed_dim)))
biseq = nn.BiSequencer(nn.FastLSTM(embed_dim, embed_dim), nn.FastLSTM(embed_dim, embed_dim), 1)
batchLSTM:add(biseq)
print(batchLSTM:forward(inputs))
print("Test")
print(nn.JoinTable(1,1):forward({torch.randn(4, 30), torch.randn(4, 30)}))
print(biseq.bwdSeq)

batchLSTM:add(nn.Sequencer(nn.Linear(2*embed_dim, output_dim)))
batchLSTM:add(nn.Sequencer(nn.LogSoftMax()))
print(batchLSTM:forward(inputs))

