import pandas as pd 
import numpy as np
#import matplotlib.pyplot as plt 

#xs = np.array(range(len(res)))
dictionary = pd.read_csv("data/words.copy.dict", sep='\t', index_col="index")
sentence = np.loadtxt("sentence_outputs.txt", skiprows=1, delimiter=",").astype(int)

res = []
for word in sentence:
	res.append(dictionary.ix[word, "word"])
print " ".join(res)


#print(dictionary)


