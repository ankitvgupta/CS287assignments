dofile(_G.path.."utils.lua")

function init_trie()
	return {}
end


function add_target_to_position(position, target)
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

	add_target_to_position(position, target)
	-- iterate backwards through the context (since this is a reverse trie)
	for i = sentence_len, 1, -1 do
		local word = context[i]

		if position[word] ~= nil then
			position = position[ word]
		else 
			position[word] = {}
			position = position[word]
		end
		add_target_to_position(position, target)
	end

end

-- Given a trie and the context being looked for, returns a table
--       with the counts of each word that proceeded that context
-- 		 Returns nil if context not in trie.
function get_word_counts_for_context(trie, context, vocab_size, alpha)

	-- Deal with the special case of the highest level position (unigrams - no previous words)
	if #context:size() == 0 then
		--return trie['counts']
		return add_to_tab(trie['counts'], vocab_size, alpha)
	end

	local context_len = context:size(1)
	--print(context_len)

	local position = trie
	-- iterate backwards through the sentence's context (this is a reverse trie)
	for i = context_len, 1, -1 do
		local word = context[i]

		if position[word] ~= nil then
			position = position[word]
		else 
			--return {}
			return add_to_tab({}, vocab_size, alpha)
		end

	end
	if position['counts'] then
		-- Explicitely add the count for the <s> string which will never be predicted.
		--position['counts'][3] = 0
		--return position['counts']
		return add_to_tab(position['counts'], vocab_size, alpha)
	end
	--return {}
	return add_to_tab({}, vocab_size, alpha)
end


-- input_contexts: Torch LongTensor (N x d_win)
-- output_words: Torch LongTensor (N x 1)
-- Both of these should use the IDs of the 
function fit(input_contexts, output_words)

	-- Make sure the inputs are valid 
	assert(input_contexts:size(1) == output_words:size(1))
	local N = input_contexts:size(1)

	-- Load data into Trie
	local reverse_trie = init_trie()
	for i = 1, N do
		add_word_and_context_to_trie(reverse_trie, input_contexts[i],output_words[i])
	end
	return reverse_trie
end

function normalize_table(tab)
	local total = sum_of_values_in_table(tab)
	return multiply_table_by_x(tab, 1/total)
end

-- Returns the distribution over the vocabulary given the context
-- Trie should be a trained trie 
-- Context is a LongTensor.
--
-- Note that this function operates recursively
function predict(trie, context, vocab_size, alpha)
	--print("Predict", vocab_size, alpha)
	local num_words = -1
	-- If there is no context, just return the base case
	if #context:size() == 0 then 
		return normalize_table(get_word_counts_for_context(trie, context, vocab_size, alpha))
	else 
		num_words = context:size(1)
	end
	--local num_words = context:size(1)
	local count_table = get_word_counts_for_context(trie, context, vocab_size, alpha)
	--count_table = add_to_tab(count_table, vocab_size, alpha)
	--count_table[3] = 0
	local F_cstar = sum_of_values_in_table(count_table)
	local N_cstar = number_of_items_in_table(count_table, alpha) 
	--print(F_cstar, N_cstar)
	assert(F_cstar > 0)

	-- This implements F_{c,w} + N_{C,star}*p_wb(w|c')
	local numerator = nil
	if num_words == 1 then -- base case, the unigram case is just the distribution of final words.
		--numerator = sum_tables(count_table, multiply_table_by_x(predict(trie, torch.LongTensor{}),N_cstar))
		numerator = sum_tables(count_table, multiply_table_by_x(predict(trie, torch.LongTensor{}, vocab_size, alpha), N_cstar))
	elseif num_words > 1 then
		numerator = sum_tables(count_table, multiply_table_by_x(predict(trie, context:narrow(1, 2, num_words - 1), vocab_size, alpha),N_cstar))
	end

	-- This implements the rest of the fraction
	local p_wb = multiply_table_by_x(numerator, 1.0/(F_cstar + N_cstar))
	return p_wb
end

function isnan(x) return x ~= x end

function predictall_and_subset(trie, valid_input, valid_options, vocab_size, alpha)
	assert(valid_input:size(1) == valid_options:size(1))
	print("Starting predictions")
	local predictions = torch.zeros(valid_input:size(1), valid_options:size(2))
	print("Initialized predictions tensor")
	for i = 1, valid_input:size(1) do
		if i % 100 == 0 then
			--print("Iteration", i, "MemUsage", collectgarbage("count")*1024)
			collectgarbage()
		end
		local prediction = table_to_tensor(predict(trie, valid_input[i], vocab_size, alpha), vocab_size)
		assert(prediction:sum() > .99999 and prediction:sum() < 1.000001)
		local values_wanted = prediction:index(1, valid_options[i])
		values_wanted:div(values_wanted:sum())
		predictions[i] = values_wanted
	end
	return predictions
end


-- Creates a simple trie that demonstrates its main features.
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

-- Creates a trie with randomly generated words
--      num_sentences: the number of sentences to insert into the trie
--      length: The length of each sentence (this assumes fixed length)
--      vocab_size: The number of words in the vocabulary
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
end



--trie_example()
--bigtrie_example(1000000,5,1000)
