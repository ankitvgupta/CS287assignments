require 'nn'
require 'rnn'
--require 'cunn'

local b = 2
local hidden = 5

local input = torch.zeros(6, 8)
input[1][1] = 1
input[2][2] = 1
input[3][1] = 1
input[4][2] = 1
input[5][1] = 1
input[6][2] = 1
input:narrow(2, 3, 6):add(torch.randn(6, 6))
input = input:reshape(b, input:size(1)/b, input:size(2))
print(input)



batchLSTM = nn.Sequential()
batchLSTM:add(nn.Transpose{1,2})
batchLSTM:add(nn.SplitTable(1,3))
print(batchLSTM:forward(input))
batchLSTM:add(nn.Sequencer(nn.Linear(8, hidden)))
batchLSTM:add(nn.BiSequencer(nn.FastLSTM(hidden,hidden), nn.FastLSTM(hidden,hidden)))
print(batchLSTM:forward(input))