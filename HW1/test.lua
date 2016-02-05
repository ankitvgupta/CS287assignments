--Testing purposes only!
require("nn")

dofile("utils.lua")
dofile("logisticregression.lua")

-- checks sparseMultiply by using convertSparseToReal and then doing normal matrix multiply
function checkSparseMultiply(X, W)
	local numClasses = W:size()[1]
	local numFeatures = W:size()[2]
	local numEntries = X:size()[1]

	local denseX = torch.zeros(numEntries, numFeatures)

	for i = 1, numEntries do
		denseX[i] = convertSparseToReal(X[i], numFeatures)
	end

	local trueAnswer = torch.mm(denseX, W:t())

	printv("Starting sparse multiply...", 3)
	local ourAnswer = sparseMultiply(X, W:t())
	printv("Done.", 3)

	local testResult = tensorsEqual(ourAnswer, trueAnswer)
	printtest("checkSparseMultiply", testResult)
end

function checkSoftmax(X, W, b)
	local m = nn.SoftMax()

	local Ans = sparseMultiply(X, W:t())
	for i = 1, Ans:size()[1] do
		Ans[i]:add(b)
	end

	local trueAnswer = m:forward(Ans)

	printv("Starting softmax...", 3)
	local ourAnswer = softmax(X, W, b)
	printv("Done.", 3)

	local testResult = tensorsEqual(ourAnswer, trueAnswer)
	printtest("checkSoftmax", testResult)
end

function checkCrossEntropy(W, b, X, Y)
	local criterion = nn.CrossEntropyCriterion()
	local z = sparseMultiply(X, W:t())

	local numEntries = z:size()[1]
	local numClasses = z:size()[2]

	for i = 1, numEntries do
		z[i]:add(b)
	end
	
	local trueAnswer = criterion:forward(z, Y)*numEntries

	printv("Starting crossEntropy...", 3)
	local ourAnswer = crossEntropy(X, W, b, Y)
	printv("Done.", 3)

	local testResult = (torch.abs(trueAnswer - ourAnswer) < ERROR_TOLERANCE)
	printtest("checkCrossEntropy", testResult)

	if not testResult then
		print("True Answer:", trueAnswer, "Our Answer:", ourAnswer)
	end
end

function checkGradient(W, b, X, Y)
	local numEntries = X:size()[1]
	local numFeatures = W:size()[2]
	local numClasses = W:size()[1]

	local denseX = torch.zeros(numEntries, numFeatures)

	for i = 1, numEntries do
		denseX[i] = convertSparseToReal(X[i], numFeatures)
	end

	local linLayer = nn.Linear(numFeatures,numClasses)
	local softMaxLayer = nn.LogSoftMax()
	local model = nn.Sequential()

	model:add(linLayer)
	model:add(softMaxLayer)

	local criterion = nn.CrossEntropyCriterion()
	
	x, dl_dx = model:getParameters()
	dl_dx:zero()
	criterion:forward(model:forward(denseX), Y)
   	model:backward(denseX, criterion:backward(model.output, Y))
	local trueAnswer = dl_dx
	trueAnswer = trueAnswer:view(numClasses, numFeatures+1)
	trueAnswerW = trueAnswer:narrow(2, 1, numFeatures)
	trueAnswerb = trueAnswer:narrow(2, numFeatures, 1):squeeze()

	printv("Starting gradient...", 3)
	local ourAnswerW, ourAnswerb = gradient(W, b, X, Y, 1, numEntries)
	printv("Done.", 3)

	local testResultW = tensorsEqual(ourAnswerW, trueAnswerW)
	local testResultb = tensorsEqual(ourAnswerb, trueAnswerb)
	local testResult = testResultW and testResultb
	printtest("checkGradient", testResult)
end


-- tests
local numEntries, numClasses, numFeatures = 1000, 5, 500
local Y = torch.Tensor(numEntries):apply(function() return torch.random(1, 5) end)
local X = (torch.rand(numEntries,numClasses):mul(numFeatures-1):abs():round() + 2):long()
local W = torch.randn(numClasses, numFeatures)
local b = torch.randn(numClasses)

checkSparseMultiply(X, W)
checkSoftmax(X, W, b)
checkCrossEntropy(W, b, X, Y)
checkGradient(W, b, X, Y)
