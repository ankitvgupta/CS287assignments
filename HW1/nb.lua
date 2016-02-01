-- Only requirement allowed
require("hdf5")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')

-- Hyperparameters
-- ...
function unitTest()
	local x = torch.ones(3, 2)
	x[1][1] = 5
	x[2][1] = 3
	x[2][2] = 2
	x[3][1] = 2
	local Wt = torch.Tensor(4, 2)
	local b = torch.ones(1, 2)
	i=0; Wt:apply(function() i = i+1; return i end)
	Wt[4][2] = 0
	local output = validateModel(Wt:t(), b, x)
	print(output)

end

function sparseMultiply(A, B)
	-- A is a sparse tensor with 1-padding
	-- B is a dense tensor
	-- Matrix multiplication A*B in the straightforward way
	numRows = A:size(1)
	numCols = B:size(2)
	local output = torch.Tensor(numRows, numCols)
	for r = 1, numRows do
		for c = 1, numCols do
			local dotProd = 0
			-- dot product of row r in dense A and col c in B
			for j = 1, A:size(2) do		
				indexIntoB = A[r][j]-1
				if indexIntoB == 0 then
					break
				else
					dotProd = dotProd + B[indexIntoB][c]
				end
			end
			output[r][c] = dotProd
		end
	end
	return output
end


-- W and b are the weights to be trained. X is the sparse matrix representation of the input. Y is the classes
function validateModel(W, b, x, y)
    Ans = sparseMultiply(x, W:t())
    for r = 1, Ans:size(1) do
    	Ans[r]:add(b)
    end
    a, b = torch.max(Ans, 2)
    equality = torch.eq(b, y)
    --print(torch.sum(equality, 1)[1])

    score = equality:sum()/equality:size()[1]
    return score
end   
    

function createCountsMatrix(training_input, training_output)
	print("     CreateCountsMatrix: Opened Data File")
	local f = hdf5.open(opt.datafile, 'r')
	local F = torch.zeros(nfeatures, nclasses)
	--local training_input = f:read(features_table):all():double()
	--local training_output = f:read(classes_table):all():double()
	print("     CreateCountsMatrix: Loaded training data")
	local train_size = training_input:size()
	for n = 1, train_size[1] do
		for j = 1, train_size[2] do
			feat = training_input[n][j] - 1
			class = training_output[n]
			if feat > 0 then
				F[feat][class] = F[feat][class] + 1
			end
		end
	end
	print("     CreateCountsMatrix: Calculated counts")
	return F
end



function naiveBayes(alpha)
	local f = hdf5.open(opt.datafile, 'r')
	local training_input = f:read('train_input'):all():double()
	local training_output = f:read('train_output'):all():double()
	--[[
	print("     NaiveBayes: Opened Data File")
	local f = hdf5.open(opt.datafile, 'r')
	local F = torch.zeros(nfeatures, nclasses)
	local training_input = f:read('train_input'):all():double()
	local training_output = f:read('train_output'):all():double()
	print("     NaiveBayes: Loaded training data")
	local train_size = training_input:size()
	for n = 1, train_size[1] do
		for j = 1, train_size[2] do
			feat = training_input[n][j] - 1
			class = training_output[n]
			if feat > 0 then
				F[feat][class] = F[feat][class] + 1
			end
		end
	end
	print("     NaiveBayes: Calculated counts")
	--]]
	local F = createCountsMatrix(training_input, training_output)

	-- Add a small offset for smoothing
	F = F + alpha

	-- Now, we column normalize the Tensor
	sum_of_each_col = torch.sum(F, 1)
	p_x_given_y = torch.Tensor(nfeatures, nclasses):zero()
	for n = 1, F:size()[1] do
		p_x_given_y[n] = torch.cdiv(F[n] , sum_of_each_col)
	end
	print("     NaiveBayes: Calculated p(x|y)")
	class_distribution = torch.zeros(nclasses)
	for n=1, training_output:size()[1] do
		class = training_output[n]
		class_distribution[class] = class_distribution[class] + 1
	end
	p_y = torch.div(class_distribution, torch.sum(class_distribution, 1)[1])
	print("     NaiveBayes: Calculated p(y)")

	local W = torch.log(p_x_given_y)
	local b = torch.log(p_y)
	local validation_input = f:read('valid_input'):all():double()
	local validation_output = f:read('valid_output'):all():long()
	print("     NaiveBayes: Initialized validation parameters")

    validation_accuracy = validateModel(W:t(), b, validation_input,validation_output)
	print("     NaiveBayes: Validation Accuracy:", validation_accuracy)

	return validation_accuracy	
end


function main() 
   -- Parse input params
   opt = cmd:parse(arg)
   local f = hdf5.open(opt.datafile, 'r')
   print("Opened datafile")
   nclasses = f:read('nclasses'):all():long()[1]
   nfeatures = f:read('nfeatures'):all():long()[1]
   print("Loaded size parameters")

   local W = torch.randn(nclasses, nfeatures)
   local b = torch.DoubleTensor(nclasses)
   local validation_input = f:read('valid_input'):all():double()
   --print(validateModel(W, b, validation_input))
   naiveBayes(1)
   --unitTest()
   -- Train.

   -- Test.
end

main()
