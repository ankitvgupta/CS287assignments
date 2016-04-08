-- Used only for softmax.
require("nn")

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
	return torch.cdiv(F, F:sum(2):expand(F:size()))
end

function col_normalize(F):
	return torch.cdiv(F, F:sum(1):expand(F:size()))
end

-- F_transition and F_emission are the matricies reduced by the transition_F_matrix() and
-- emission_F_matrix() functions above, with laplace smoothing added.
-- This function returns a predictor function, which takes c_{i-1} and x_i and returns the prob distribution for c_i.
function make_predictor_function(F_transition, F_emission)
	local emission_p_y = F_emission:sum(2)/F_emission:sum()
	local emission_p_x_given_y = row_normalize(F_emission)
	local emission_joint = torch.cmul(p_x_given_y, p_y:expand(p_x_given_y:size()))
	local emission_p_y_given_x = col_normalize(joint):t()
	local log_emission_p_y_given_x = torch.log(emission_p_y_given_x)

	local transition_p_y_given_yprev = row_normalize(F_transition)
	local log_transition_p_y_given_yprev = torch.log(transition_p_y_given_yprev)

	local softmax_layer = nn.SoftMax()
	local predictor = function (c_prev, x_i) return softmax_layer:forward(log_transition_p_y_given_yprev[c_prev] + log_emission_p_y_given_x[x_i]) end			
	return predictor
end