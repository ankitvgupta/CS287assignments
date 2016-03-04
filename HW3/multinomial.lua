function init_trie()
	return {}
end

-- trie is the trie we are adding to (note that this is passed by reference)
-- sentence is a torch tensor consisting of the words
function add_word_and_context_to_trie(trie, sentence)
	local sentence_len = sentence:size(1)

	-- The last word is one being predicted
	local target = sentence[sentence_len]

	local position = trie
	-- iterate backwards through the sentence's context (this is a reverse trie)
	for i = sentence_len - 1, 1, -1 do
		local word = sentence[i]

		if position[word] ~= nil then
			position = position[word]
			--position['count']  = position['count'] + 1
		else 
			position[word] = {}
			position = position[word]
			--position['count'] = 1
		end

	end
	if position['counts'] == nil then
		local counts = {}
		counts[target] = 1
		position['counts'] = counts
	elseif position['counts'][target] == nil then
		position['counts'][target] = 1
	else
		position['counts'][target] = position['counts'][target]  + 1
	end 

end

-- Given a trie and the context being looked for, returns a table
--       with the counts of each word that proceeded that context
-- 		 Returns nil if context not in trie.
function get_word_counts_for_context(trie, context)

	local context_len = context:size(1)

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
	return position['counts']
end

function normalize_table(tab)
	local total = 0
	for _, val in pairs(tab) do
		total = total + val
	end
	new_table = {}
	for key, val in pairs(tab) do
		new_table[key] = val/total
	end
	return new_table
end


function trie_example()
	local reverse_trie = init_trie()
	add_word_and_context_to_trie(reverse_trie, torch.LongTensor{1,2,3})
	add_word_and_context_to_trie(reverse_trie, torch.LongTensor{2,2,3})
	add_word_and_context_to_trie(reverse_trie, torch.LongTensor{1,1,3})
	add_word_and_context_to_trie(reverse_trie, torch.LongTensor{2,1,3})
	add_word_and_context_to_trie(reverse_trie, torch.LongTensor{1,1,3})
	add_word_and_context_to_trie(reverse_trie, torch.LongTensor{1,1,3})
	add_word_and_context_to_trie(reverse_trie, torch.LongTensor{1,1,2})
	add_word_and_context_to_trie(reverse_trie, torch.LongTensor{1,3})
	print(reverse_trie)
	local counts = get_word_counts_for_context(reverse_trie, torch.LongTensor{1,1})
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
    while counts == nil do
        try = try + 1
        print(try)
	    counts = get_word_counts_for_context(reverse_trie, torch.rand(length):mul(vocab_size):long())
    end
    print(counts)
    
--	print(reverse_trie)
--	local counts = get_word_counts_for_context(reverse_trie, torch.LongTensor{1,1})
--	print(counts)
--	print(normalize_table(counts))
end

--trie_example()
bigtrie_example(1000000,5,10000)
