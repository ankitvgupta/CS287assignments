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