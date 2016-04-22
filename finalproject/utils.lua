function reshape_inputs(num_sequences, input, output)
	assert(input:size(1) == output:size(1))
	local len_wanted = input:size(1) - (input:size(1) % num_sequences)
	local new_inputs = input:narrow(1, 1, len_wanted):reshape(num_sequences, len_wanted/num_sequences)
	local new_outputs = output:narrow(1, 1, len_wanted):reshape(num_sequences, len_wanted/num_sequences)
	return new_inputs, new_outputs
end