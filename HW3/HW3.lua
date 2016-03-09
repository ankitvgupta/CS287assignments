-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-testfile', '', 'test file')
cmd:option('-save_losses', '', 'file to save loss per epoch to (leave blank if not wanted)')
cmd:option('-classifier', 'nn', 'classifier to use (mle, nn)')
cmd:option('-alpha', 1, 'laplacian smoothing factor')
cmd:option('-eta', .1, 'Learning rate (.1 for adagrad, 500 for sgd, 10 for nn sgd)')
cmd:option('-lambda', 0, 'regularization penalty (not implemented)')
cmd:option('-minibatch', 2000, 'Minibatch size (500 for nn, 2000 for lr)')
cmd:option('-epochs', 20, 'Number of epochs of SGD')
cmd:option('-optimizer', 'sgd', 'Name of optimizer to use (adagrad or sgd)')
cmd:option('-hiddenlayers', 10, 'Number of hidden layers (if using neural net)')
cmd:option('-embedding_size', 50, 'Size of word embedding')
cmd:option('-odyssey', false, 'Set to true if running on odyssey')
cmd:option('-K', 10, 'for NCE only')

-- Hyperparameters
-- ...


function main() 
   -- Parse input params
   opt = cmd:parse(arg)

   --print("Datafile:", opt.datafile, "Classifier:", opt.classifier, "Alpha:", opt.alpha, "Eta:", opt.eta, "Lambda:", opt.lambda, "Minibatch size:", opt.minibatch, "Num Epochs:", opt.epochs, "Optimizer:", opt.optimizer, "Hidden Layers:", opt.hiddenlayers, "Embedding size:", opt.embedding_size)


   _G.path = opt.odyssey and '/n/home09/ankitgupta/CS287/CS287assignments/HW3/' or ''

   dofile(_G.path..'train.lua')
   dofile(_G.path..'multinomial.lua')
   dofile(_G.path..'utils.lua')

   printoptions(opt)

   local f = hdf5.open(opt.datafile, 'r')
   --local samples = nil
   local f2 = hdf5.open(_G.path..'samples.hdf5')
   local samples = f2:read("samples"):all():long()
   print("Sample dist", torch.min(samples), torch.max(samples))

   local nclasses = f:read('numClasses'):all():long()[1]
   local nfeatures = f:read('numFeatures'):all():long()[1]
   local d_win = f:read('d_win'):all():long()[1]

   print("nclasses:", nclasses, "nfeatures:", nfeatures, "d_win:", d_win)

   local training_input = f:read('train_input'):all():long()   
   local training_output = f:read('train_output'):all():long()

   local valid_input = f:read('valid_input'):all():long()
   local valid_output = f:read('valid_output'):all():long()
   print("Full valid size", valid_input:size(), valid_output:size())


   local valid_blanks_input = f:read('valid_blanks_input'):all():long()
   local valid_blanks_options = f:read('valid_blanks_options'):all():long()
   local valid_blanks_outputs = f:read('valid_blanks_output'):all():long()

   local test_blanks_input = f:read('test_blanks_input'):all():long()
   local test_blanks_options = f:read('test_blanks_options'):all():long()

   local result

   -- Train neural network.
   if opt.classifier == "nn" then
   		local model, criterion = neuralNetwork(nfeatures, opt.hiddenlayers, nclasses, opt.embedding_size, d_win)
   		model = trainModel(model, criterion, training_input, training_output, valid_blanks_input, valid_blanks_options, valid_blanks_outputs, opt.minibatch, opt.epochs, opt.optimizer, opt.save_losses)
		  local acc, cross_entropy_loss = getaccuracy2(model, valid_blanks_input, valid_blanks_options, valid_blanks_outputs)
		--result = predictall_and_subset(model, valid_input, valid_options, nclasses, opt.alpha)
		--local acc = get_result_accuracy(result, valid_input, valid_options, valid_true_outs)
      local full_cross_ent = NNLM_CrossEntropy(model, valid_input, valid_output)
    	printoptions(opt)
    	print("Results (Accuracy,SmallCrossEntropy,SmallPerp, FullCrossEntropy, Fullperp):", acc, cross_entropy_loss, torch.exp(cross_entropy_loss), full_cross_ent, torch.exp(full_cross_ent))

    	--print(acc)

		--print("Accuracy:")
		--print(getaccuracy(model, valid_input, valid_options, valid_true_outs))
	elseif opt.classifier == 'nce' then
    local reverse_trie = fit(training_input, training_output)
    local distribution = normalize_table(get_word_counts_for_context(reverse_trie, torch.LongTensor{}, nclasses, opt.alpha))
    local p_ml_tensor = table_to_tensor(distribution, nclasses)

		local model, lookup, bias = trainNCEModel(training_input, training_output,
					valid_blanks_input, 
					valid_blanks_options,
					valid_blanks_outputs,
					opt.minibatch, 
					opt.epochs, 
					opt.optimizer, 
					opt.save_losses,
					nfeatures, opt.hiddenlayers, nclasses, opt.embedding_size, d_win, opt.alpha, opt.eta, samples, opt.K, p_ml_tensor,valid_input, valid_output)

    local acc, cross_entropy_loss, perplexity = getNCEStats(model, lookup, bias, valid_blanks_input, valid_blanks_options, valid_blanks_outputs, p_ml_tensor)

    --local predictions = NCE_predictions(model, lookup, bias, valid_input, valid_options)
    --print(predictions:sum(2))
    --local acc,cross_entropy_loss = get_result_accuracy(predictions, valid_input, valid_options, valid_true_outs), cross_entropy_loss(valid_true_outs, predictions, valid_options)
    -- Combine the models to a normal nn model for making predictions
    --local prediction_model = make_NCEPredict_model(model, lookup, bias, opt.hiddenlayers, nclasses)
    --local acc, cross_entropy_loss = getaccuracy2(prediction_model, valid_input, valid_options, valid_true_outs)
    local full_cross_ent = NCE_predictions2(model, lookup, bias, valid_input, valid_output, opt.hiddenlayers, nclasses)
    printoptions(opt)
    print("Results(Acc,Cross,Perp,FullCross,FullPerp):", acc, cross_entropy_loss, perplexity, full_cross_ent, torch.exp(full_cross_ent) )


    elseif opt.classifier == 'multinomial' then
    	local reverse_trie = fit(training_input, training_output)
    	--print(get_word_counts_for_context(reverse_trie, torch.LongTensor{}, nclasses, opt.alpha))
		  local predicted_distributions = predictall_and_subset(reverse_trie, valid_blanks_input, valid_blanks_options, nclasses, opt.alpha)
    	--print(predicted_distributions:sum(2))
    	local cross_entropy_loss = cross_entropy_loss(valid_blanks_outputs, predicted_distributions, valid_blanks_options)
    	print("Cross-entropy loss", cross_entropy_loss)

    	--result = predictall_and_subset(reverse_trie, test_context, test_options, nclasses, opt.alpha)

    	local acc = get_result_accuracy(predicted_distributions, valid_blanks_input, valid_blanks_options, valid_blanks_outputs)
    	printoptions(opt)
    	print("Results:", acc, cross_entropy_loss, torch.exp(cross_entropy_loss))
    elseif opt.classifier == 'laplace' then
      local reverse_trie = fit(training_input, training_output)
      --print(get_word_counts_for_context(reverse_trie, torch.LongTensor{}, nclasses, opt.alpha))
      local predicted_distributions = getlaplacepredictions(reverse_trie, valid_blanks_input, valid_blanks_options, nclasses, opt.alpha)
      --print(predicted_distributions:sum(2))
      local cross_entropy_loss = cross_entropy_loss(valid_blanks_outputs, predicted_distributions, valid_blanks_options)
      print("Cross-entropy loss", cross_entropy_loss)

      --result = predictall_and_subset(reverse_trie, test_context, test_options, nclasses, opt.alpha)

      local acc = get_result_accuracy(predicted_distributions, valid_blanks_input, valid_blanks_options, valid_blanks_outputs)
      printoptions(opt)
      print("Results:", acc, cross_entropy_loss, torch.exp(cross_entropy_loss))
    elseif opt.classifier == 'mle' then
      local reverse_trie = fit(training_input, training_output)
      --print(get_word_counts_for_context(reverse_trie, torch.LongTensor{}, nclasses, opt.alpha))
      local predicted_distributions = getmlepredictions(reverse_trie, valid_blanks_input, valid_blanks_options, nclasses, opt.alpha)
      --print(predicted_distributions:sum(2))
      local cross_entropy_loss = cross_entropy_loss(valid_blanks_outputs, predicted_distributions, valid_blanks_options)
      print("Cross-entropy loss", cross_entropy_loss)

      --result = predictall_and_subset(reverse_trie, test_context, test_options, nclasses, opt.alpha)

      local acc = get_result_accuracy(predicted_distributions, valid_blanks_input, valid_blanks_options, valid_blanks_outputs)
      printoptions(opt)
      print("Results:", acc, cross_entropy_loss, torch.exp(cross_entropy_loss))
    else
    	print("Error: classifier '", opt.classifier, "' not implemented")
    end

    if (opt.testfile ~= '') then
        --print("Writing to test file")
    	write_predictions(result, opt.testfile)
    end





end

main()
