

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

reverse_trie = init_trie()
add_word_and_context_to_trie(reverse_trie, torch.LongTensor{1,2,3})
add_word_and_context_to_trie(reverse_trie, torch.LongTensor{2,2,3})
add_word_and_context_to_trie(reverse_trie, torch.LongTensor{1,1,3})
add_word_and_context_to_trie(reverse_trie, torch.LongTensor{2,1,3})
add_word_and_context_to_trie(reverse_trie, torch.LongTensor{1,1,3})
add_word_and_context_to_trie(reverse_trie, torch.LongTensor{1,1,3})
add_word_and_context_to_trie(reverse_trie, torch.LongTensor{1,1,2})

print(reverse_trie)