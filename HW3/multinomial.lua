function init_trie()
	return {}
end

-- trie is the trie we are adding to (note that this is passed by reference)
-- sentence is a torch tensor consisting of the words
-- The target is optional - if included, then whole window used as context. Otherwise, 
--      last element of context is the sentence
function add_word_and_context_to_trie(trie, context, target)
	local sentence_len = context:size(1)

	if target == nil then
		target = context[sentence_len]
		sentence_len = sentence_len - 1
	end

	-- The last word is one being predicted
	--local target = context[sentence_len]

	local position = trie
	-- iterate backwards through the context (since this is a reverse trie)
	for i = sentence_len, 1, -1 do
		local word = context[i]

		if position[word] ~= nil then
			position = position[word]
		else 
			position[word] = {}
			position = position[word]
		end

		-- If this is the first time we have gotten here.
		if position['counts'] == nil then
			local counts = {}
			counts[target] = 1
			position['counts'] = counts
		-- If we have gotten here, but the target is different
		elseif position['counts'][target] == nil then
			position['counts'][target] = 1
		-- If we have gotten here and seen this target before
		else
			position['counts'][target] = position['counts'][target]  + 1
		end 
	end

end

-- Given a trie and the context being looked for, returns a table
--       with the counts of each word that proceeded that context
-- 		 Returns nil if context not in trie.
function get_word_counts_for_context(trie, context)

	local context_len = context:size(1)
	--print(context_len)

	local position = trie
	-- iterate backwards through the sentence's context (this is a reverse trie)
	for i = context_len, 1, -1 do
		local word = context[i]

		if position[word] ~= nil then
			position = position[word]
		else 
			return nil
		end

	end
	--print(position)
	return position['counts']
end

-- This can be used to calculate N_{c,*}
function number_of_items_in_table(tab)
	return #tab
end

-- This can be used to calculate F_{c,*}
function sum_of_values_in_table(tab)
	local total = 0
	for _, val in pairs(tab) do
		total = total + val
	end
	return total
end

function multiply_table_by_x(tab, x)
	local new_table = {}
	for key, val in pairs(tab) do
		new_table[key] = val*x
	end
	return new_table
end

function sum_tables(tab1, tab2):
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


-- input_contexts: Torch LongTensor (N x d_win)
-- output_words: Torch LongTensor (N x 1)
-- Both of these should use the IDs of the 
function train(input_contexts, output_words)

	-- Make sure the inputs are valid 
	assert(input_contexts:size(1) == output_words:size(1))
	local N = input_contexts:size(1)

	-- Load data into Trie
	local reverse_trie = init_trie()
	for i in 1, N do
		add_word_and_context_to_trie(reverse_trie, input_contexts[i],output_words[i])
	end
	return reverse_trie
end

-- function p_ML(count_table)
-- 	return normalize_table(count_table)
-- end

-- Returns the distribution over the vocabulary given the context
-- Trie should be a trained trie 
-- Context is a LongTensor.
--
-- Note that this function operates recursively
function predict(trie, context)
	local num_words = context:size(1)
	if num_words == 0 then
		return {}
	end
	local count_table = get_word_counts_for_context(trie, context)
	local F_cstar = sum_of_values_in_table(count_table)
	local N_cstar = number_of_items_in_table(count_table) 

	-- This implements F_{c,w} + N_{C,star}*p_wb(w|c')
	local numerator = sum_tables(count_table, multiply_table_by_x(predict(trie, context:narrow(1, 2, num_words - 1)),N_cstar))

	-- This implements the rest of the fraction
	local p_wb = multiply_table_by_x(numerator, 1.0/(F_cstar + N_cstar))
	return p_wb
end

function normalize_table(tab)
	local total = sum_of_values_in_table(tab)
	return multiply_table_by_x(tab, 1/total)
	-- local new_table = {}
	-- for key, val in pairs(tab) do
	-- 	new_table[key] = val/total
	-- end
	-- return new_table
end


function trie_example()
	local reverse_trie = init_trie()
	add_word_and_context_to_trie(reverse_trie, torch.LongTensor{1,2,3},1)
	add_word_and_context_to_trie(reverse_trie, torch.LongTensor{2,2,3},2)
	add_word_and_context_to_trie(reverse_trie, torch.LongTensor{1,1,3},2)
	add_word_and_context_to_trie(reverse_trie, torch.LongTensor{2,1,3},2)
	add_word_and_context_to_trie(reverse_trie, torch.LongTensor{1,1,3},1)
	add_word_and_context_to_trie(reverse_trie, torch.LongTensor{1,1,3},1)
	add_word_and_context_to_trie(reverse_trie, torch.LongTensor{1,1,2},1)
	add_word_and_context_to_trie(reverse_trie, torch.LongTensor{1,3},1)
	print(reverse_trie)
	local counts = get_word_counts_for_context(reverse_trie, torch.LongTensor{1,3})
	print(counts)
	print(normalize_table(counts))
end

-- Creates a trie with large number of 
function bigtrie_example(num_sentences, length, vocab_size)
	local reverse_trie = init_trie()
	for i = 1, num_sentences do
		if i % 10000 == 0 then
			print(i)
		end
		add_word_and_context_to_trie(reverse_trie, torch.rand(length):mul(vocab_size):long())
	end
    print("Getting counts")
    local try = 0
    local counts = nil
    print(get_word_counts_for_context(reverse_trie, torch.LongTensor{1}))
    print(get_word_counts_for_context(reverse_trie, torch.LongTensor{2}))
    print(get_word_counts_for_context(reverse_trie, torch.LongTensor{3}))

    --print(reverse_trie[1]['counts'])
    --while counts == nil do
    --	try = try + 1
    --	print(try)
	--	counts = get_word_counts_for_context(reverse_trie, torch.LongTensor{try})
    --end
    --print(counts)
    
--	print(reverse_trie)
--	local counts = get_word_counts_for_context(reverse_trie, torch.LongTensor{1,1})
--	print(counts)
--	print(normalize_table(counts))
end



trie_example()
--bigtrie_example(1000000,5,1000)
