dofile("utils.lua")

-- checks sparseMultiply by using convertSparseToReal and then doing normal matrix multiply
function checkSparseMultiply(numEntries, numClasses, numFeatures)
	xtmp = (torch.rand(numEntries,numClasses):mul(numFeatures-1):abs():round() + 2):long()
	newarray = torch.zeros(xtmp:size()[1], numFeatures)

	for i = 1, newarray:size()[1] do
		newarray[i] = convertSparseToReal(xtmp[i], numFeatures)
	end
	local B = torch.randn(numFeatures, numClasses)

	local ourAnswer = torch.mm(newarray, B)

	printv("Starting sparse multiply...", 3)
	local trueAnswer = sparseMultiply(xtmp, B)
	printv("Done.", 3)

	printv(xtmp, 4)
	printv(newarray, 4)
	printv(ourAnswer, 4)
	printv(trueAnswer, 4)

	if (ourAnswer:eq(trueAnswer):sum() - numClasses*numEntries) < 1 then
		print("Test passed.", 0)
	else
		print ("Test failed.", 0)
	end
end

-- tests
checkSparseMultiply(1000, 5, 50)