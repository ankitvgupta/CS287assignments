-- Only requirement allowed
require("hdf5")

-- Common functions and classifiers
dofile("nb.lua")
dofile("logisticregression.lua")
dofile("hinge.lua")
dofile("utils.lua")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')
cmd:option('-alpha', 1, 'laplacian smoothing factor for NB')
cmd:option('-eta', .5, 'Learning rate')
cmd:option('-lambda', 1, 'regularization penalty')
cmd:option('-minibatch', 500, 'Minibatch size')
cmd:option('-epochs', 50, 'Number of epochs of SGD')
cmd:option('-min_sentence_length', 0, 'Minimum length of sentence to be included in training set')
cmd:option('-test_file', '', 'File to put results from test set. Leave nil if not wanted')
cmd:option('-generate_validation_set', 0, "Set to 1 if validation set needs to be self generated")

-- NOTE: THIS DOESNT WORK YET
function crossValidation(Xs, Ys, K, options)
	-- Generates a random ordering of rows
	local order = torch.randperm(Xs:size()[1]):long()

	-- Order the rows according to that new ordering
	local X = Xs:index(1, order)
	local Y = Ys:index(1, order)

	local set_size = math.floor(X:size()[1]/K)
	for loc = 1, 1 + (K-1)*set_size, set_size do
		X_wanted = X:narrow(1,loc,set_size)
	end

end

function printoptions(opt)
    print("Datafile:", opt.datafile, "Classifier:", opt.classifier, "Alpha:", opt.alpha, "Eta:", opt.eta, "Lambda:", opt.lambda, "Minibatch size:", opt.minibatch, "Num Epochs:", opt.epochs, "Minimum Sentence Length:", opt.min_sentence_length)
end

function split_test_train(X_vals, Y_vals, train_ratio)
    local total_size = X_vals:size()[1]
    local cutoff = total_size*train_ratio
    local train_in = X_vals:index(1, torch.range(1, cutoff):long())
    local train_out = Y_vals:index(1, torch.range(1, cutoff):long())
    local test_in = X_vals:index(1, torch.range(cutoff+1, total_size):long())
    local test_out = Y_vals:index(1, torch.range(cutoff+1, total_size):long())
    return train_in, train_out, test_in, test_out
end

function main() 
   	-- Parse input params
   	opt = cmd:parse(arg)
   	printoptions(opt)
    print("Note that not all parameters may apply to all classifiers")
    -- Load datafiles
   	printv("Loading datafiles...", 2)
   	local f = hdf5.open(opt.datafile, 'r')
   	local training_input = f:read('train_input'):all():long()
   	local training_output = f:read('train_output'):all():long()
    local validation_input = torch.Tensor()
    local validation_output = torch.Tensor()
   	print("Number of training samples before removing small sentences", training_input:size()[1])
   	training_input, training_output = removeSmallSentences(training_input, training_output, opt.min_sentence_length)
   	print("Number of training samples after removing small sentences", training_input:size()[1])
    if opt.generate_validation_set == 0 then
     	validation_input = f:read('valid_input'):all():long()
     	validation_output = f:read('valid_output'):all():long()
    else
      training_input, training_output, validation_input, validation_output = split_test_train(training_input, training_output, .9)
   	end
    local nfeatures = f:read('nfeatures'):all():long()[1]
   	local nclasses = f:read('nclasses'):all():long()[1]
   	printv("Done.", 2)

   	-- Train.
   	printv("Training classifier...", 2)
   	if opt.classifier == 'nb' then
   		W, b = naiveBayes(training_input, training_output, nfeatures, nclasses, opt.alpha)	
   	elseif (opt.classifier == 'lr') then
   		W, b = logisticRegression(training_input, training_output, validation_input, validation_output, nfeatures, nclasses, opt.minibatch, opt.eta, opt.lambda, opt.epochs)
   	elseif (opt.classifier == 'hinge') then
   		W, b = hingeLogisticRegression(training_input, training_output, validation_input, validation_output, nfeatures, nclasses, opt.minibatch, opt.eta, opt.lambda, opt.epochs)
   	else
   		printv("Unknown classifier.", 0)
   	end
   	printv("Done.", 2)

   -- Test.
   printv("Testing on validation set...", 2)
   local validation_accuracy = validateLinearModel(W, b, validation_input, validation_output)
   printv("Done", 2)
   printoptions(opt)
   print("Accuracy:", validation_accuracy)

   if (opt.test_file ~= '') then
   	    local test_input = f:read('test_input'):all():long()
        local results = torch.squeeze(getLinearModelPredictions(W, b, test_input))
        print(results)
        file = io.open(opt.test_file, 'a')
        io.output(file)
        for test_i = 1, results:size()[1] do
            io.write(test_i, ',', results[test_i], '\n')
        end
   end
   return validation_accuracy
end

main()
