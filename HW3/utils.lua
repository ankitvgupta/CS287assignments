function getaccuracy(model, validation_input, validation_options, validation_true_outs)
	local scores = model:forward(validation_input)
	local n = validation_input:size(1)
	local option_count = validation_options:size(2)

	local total_acc = 0.0

	for i=1, n do
		-- e.g. [2, 502, ..., ], where index is index of possible word
		local options = validation_options[i]
		local option_probs = {}
		local true_idx = validation_true_outs[i]

		for j=1, option_count do
			local idx = options[j]
			local s = scores[i][idx]
			if option_probs[idx] ~= nil then
				option_probs[idx] = option_probs[idx]+s
			else
				option_probs[idx] = s
			end
		end

		option_probs = normalize_table(option_probs)

		local acc = option_probs[true_idx]
		total_acc = total_acc + acc
	end

	return total_acc/n
end

function get_result_accuracy(result, validation_input, validation_options, validation_true_outs)
	local n = validation_input:size(1)
	local option_count = validation_options:size(2)

	local total_acc = 0.0

	for i=1, n do
		local options = validation_options[i]
		local option_probs = result[i]:add(-1*result[i]:min())/(result[i]:max()-result[i]:min())
		local true_idx = validation_true_outs[i]
		local acc_updated = false

		for j=1, option_count do
			if validation_options[i][j] == true_idx then
				acc = option_probs[j]
				acc_updated = true
				break
			end
		end

		assert(acc_updated)
		total_acc = total_acc + acc
	end

	return total_acc/n
end

function scores_to_preds(scores)
	_, class_preds =  torch.max(scores, 2)
	print(class_preds)
	local preds = class_preds:squeeze()
	binary_preds = torch.zeros(scores:size(1), scores:size(2))
	for i=1, scores:size(1) do
		binary_preds[i][preds[i]] = 1
	end
	return binary_preds
end

function get_predictions_from_model(model, test_input, test_options)
	local n = test_input:size(1)
	local option_count = test_options:size(2)
	local results = torch.zeros(n, option_count)

	local scores = model:forward(test_input)

	for i=1, n do
		local max_score = scores[i][1]
		local pred = 1

		for j=1, option_count do
			local idx = test_options[i][j]
			local s = scores[i][idx]
			if s > max_score then
				max_score = s
				pred = j
			end
		end

		results[i][pred] = 1
	end

	return results
end

function write_predictions(results, outfile)
	io.output(outfile)
	io.write("ID,Class1,Class2,Class3,Class4,Class5,Class6,Class7,Class8,Class9,Class10,Class11,Class12,Class13,Class14,Class15,Class16,Class17,Class18,Class19,Class20,Class21,Class22,Class23,Class24,Class25,Class26,Class27,Class28,Class29,Class30,Class31,Class32,Class33,Class34,Class35,Class36,Class37,Class38,Class39,Class40,Class41,Class42,Class43,Class44,Class45,Class46,Class47,Class48,Class49,Class50\n")
	for test_i = 1, results:size(1) do
		io.write(test_i)
		for binary_i = 1, results:size(2) do
			io.write(',', results[test_i][binary_i])
		end
		io.write('\n')
	end

end

function find(tensor_array, number)
	for i = 1, tensor_array:size(1) do
		--print(tensor_array[i])
		if tensor_array[i] == number then
			return i
		end
	end
	return -1
end

-- Calculates cross-entropy loss. This is the sum of the log
-- probabilities that were predicted for the true class.
function cross_entropy_loss(true_outputs, predicted_distribution, options)
	local logged_probabilities = torch.log(predicted_distribution)
	local loss = 0.0
	for i = 1, true_outputs:size(1) do
		predicted_distribution_index = find(options[i], true_outputs[i])
		--print(i, true_outputs[i], predicted_distribution_index, options[i])
		assert(predicted_distribution_index ~= -1)
		loss = loss + logged_probabilities[i][predicted_distribution_index]
	end
	return loss
end



-- This function might not actually have any value, but
--      it basically just normalizes the counts in a table.
function normalize_table(tab)
	local total = sum_of_values_in_table(tab)
	return multiply_table_by_x(tab, 1/total)
end

-- Calculates the number of items in a table
-- This can be used to calculate N_{c,*}
function number_of_items_in_table(tab)
	return #tab
end

-- Sums of the values in a table
-- This can be used to calculate F_{c,*}
function sum_of_values_in_table(tab)
	local total = 0
	for _, val in pairs(tab) do
		total = total + val
	end
	return total
end

-- For some reason the Torch constructor didnt work.
function table_to_tensor(tab, size)
	local t = torch.zeros(size)
	for k,v in pairs(tab) do
		t[k] = v
	end
	return t
end

-- Adds 'amount' to each key of tab from 1 to max_possible
-- This is used for laplace smoothing
function add_to_tab(tab, max_possible, amount)
	local new_tab = {}
	--print(max_possible)
	for i = 1, max_possible do
		if tab[i] ~= nil then
			new_tab[i] = tab[i] + amount
		else
			new_tab[i] = amount
		end
	end
	return new_tab
end



-- Multiply each value in a table by x
function multiply_table_by_x(tab, x)
	local new_table = {}
	for key, val in pairs(tab) do
		new_table[key] = val*x
	end
	return new_table
end

-- Takes the elementwise sum of two tables
-- This is essentially an outer join, where we sum values on duplicate keys.
function sum_tables(tab1, tab2)
	new_table = {}
	for key,val in pairs(tab1) do
		new_table[key] = val
	end
	for key, val in pairs(tab2) do
		if new_table[key] ~= nil then
			new_table[key] = new_table[key] + val
		else
			new_table[key] = val
		end
	end
	return new_table
end