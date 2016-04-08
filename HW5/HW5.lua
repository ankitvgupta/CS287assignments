-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'hmm', 'classifier to use')
cmd:option('-alpha', 1, 'laplacian smoothing factor')
cmd:option('-beta', 1, 'F score parameter')
cmd:option('-odyssey', false, 'Set to true if running on odyssey')
cmd:option('-testfile', '', 'test file (must be HDF5)')

-- Hyperparameters
-- ...


function main() 
	-- Parse input params
	opt = cmd:parse(arg)

	_G.path = opt.odyssey and '/n/home09/ankitgupta/CS287/CS287assignments/HW5/' or ''

	dofile(_G.path..'utils.lua')
	dofile(_G.path..'hmm.lua')

	local f = hdf5.open(opt.datafile, 'r')

	local nclasses = f:read('numClasses'):all():long()[1]
	local nsparsefeatures = f:read('numSparseFeatures'):all():long()[1]
	local ndensefeatures = f:read('numDenseFeatures'):all():long()[1]
	local start_class = f:read('startClass'):all():long()[1]

	local o_class = 1

	print("nclasses:", nclasses, "nsparsefeatures:", nsparsefeatures, "ndensefeatures:", ndensefeatures)

	local sparse_training_input = f:read('train_sparse_input'):all():long()
	local dense_training_input = f:read('train_dense_input'):all():long()
	local training_output = f:read('train_output'):all():long()

	local sparse_validation_input = f:read('valid_sparse_input'):all():long()
	local dense_validation_input = f:read('valid_dense_input'):all():long()
	local validation_output = f:read('valid_output'):all():long()
	local word_embeddings = f:read('word_embeddings'):all():double() 

	local sparse_test_input = f:read('test_sparse_input'):all():long()
	local dense_test_input = f:read('test_dense_input'):all():long()

	if opt.classifier == "hmm" then
		local predictor = hmm_train(sparse_training_input:squeeze(), training_output, nsparsefeatures, nclasses, opt.alpha)
		
		print("Starting Viterbi on validation set...")
		local valid_predicted_output = viterbi(sparse_validation_input:squeeze(), predictor, nclasses, start_class)
		
		print("Done. Converting to Kaggle-ish format...")
		local valid_true_kaggle, ms, mc, s = kagglify_output(validation_output, start_class, o_class)
		local valid_pred_kaggle, _, _, _ = kagglify_output(valid_predicted_output, start_class, o_class, ms, mc, s)

		print("Done. Computing statistics...")
		local f_score = compute_f_score(opt.beta, valid_true_kaggle, valid_pred_kaggle)
		print("F-Score:", f_score)

		if (opt.testfile ~= '') then
			print("Starting Viterbi on test set...")
			local test_predicted_output = viterbi(sparse_test_input:squeeze(), predictor, nclasses, start_class)

			print("Done. Converting to Kaggle-ish format...")
			local test_pred_kaggle, _, _, _ = kagglify_output(test_predicted_output, start_class, o_class)
		
			print("Done. Writing out to HDF5...")
			local f = hdf5.open(opt.testfile, 'w')
			f:write('test_outputs', test_pred_kaggle:long())
			print("Done. Wrote to ", opt.testfile, ".")

		end
			
		-- for i=1, validation_output:size(1) do
		-- 	print(i, valid_predicted_output[i], validation_output[i])
		-- end

	else
		print("error: ", opt.classifier, " is not implemented!")
	end


end

main()
