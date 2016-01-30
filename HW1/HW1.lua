-- Only requirement allowed
require("hdf5")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')

-- Hyperparameters
-- ...

function validateModel(W, b)
    local f = hdf5.open(opt.datafile, 'r')
    local validation_input = f:read('valid_input'):all():double()
    n, nfeat = validation_input:size()   
    --print(n[1], n[2], nfeat)
    --print(nclasses)
    Ans = torch.Tensor(n[1], nclasses)
    Ans:mm(validation_input, W:t())

end   
    


function main() 
   -- Parse input params
   opt = cmd:parse(arg)
   local f = hdf5.open(opt.datafile, 'r')
   nclasses = f:read('nclasses'):all():long()[1]
   nfeatures = f:read('nfeatures'):all():long()[1]

   local W = torch.DoubleTensor(nclasses, nfeatures)
   local b = torch.DoubleTensor(nclasses)
   validateModel(W, b)
   -- Train.

   -- Test.
end

main()
