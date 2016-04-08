




function make_predictor(model)
	return (function(a, b) return model + a + b end)
end

x = torch.Tensor{1,2,3}

predictor = make_predictor(x)
x:add(1)
print(predictor(1, 2))
