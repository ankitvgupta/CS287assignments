function getaccuracy(model, validation_input, validation_options, validation_true_outs)
	local scores = model:forward(validation_input)
	local n = validation_input:size(1)
	local option_count = validation_options:size(2)

	local total_acc = 0.0

	for i=1, n do
		-- e.g. [2, 502, ..., ], where index is index of possible word
		local options = validation_options[i]
		local option_probs = torch.DoubleTensor(option_count)
		local normalizer = 0.0
		local true_idx = 0 -- must be set below
		local t = validation_true_outs[i]

		for j=1, option_count do
			local idx = options[j]
			local s = scores[i][idx]
			option_probs[j] = s
			normalizer = normalizer + s
			if t == idx then
				true_idx = j
			end
		end

		option_probs:div(normalizer)
		local acc = option_probs[true_idx]
		total_acc = total_acc + acc
	end

	return total_acc/n
end
