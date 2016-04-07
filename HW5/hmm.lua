

function transition_F_matrix(classes_vector, numclasses)
	local F = torch.zeros(numclasses, numclasses)
	for i = 1, classes_vector:size(1)-1 do
		F[classes_vector[i]][classes_vector[i+1]] = F[classes_vector[i]][classes_vector[i+1]] + 1
	return F
end	


function emission_F_matrix(words, classes, numwords, numclasses)
	assert(words:size(1) == classes:size(1))
	local F = torch.zeros(numclasses, numwords)
	for i = 1, words:size(1) do
		F[classes[i]][words[i]] = F[classes[i]][words[i]] + 1
	end
	return F 
end

function row_normalize(F):
	return F:cdiv(F:sum(2):expand(F:size()))
end
