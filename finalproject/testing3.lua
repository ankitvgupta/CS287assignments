require 'nn'
require("hdf5")

local f = hdf5.open("FILT_CB513_3.hdf5", 'r')
local flat_train_input = f:read('train_input'):all():long()
print(flat_train_input:size())

print(flat_train_input[1])
print(flat_train_input[2])
print(flat_train_input[3])