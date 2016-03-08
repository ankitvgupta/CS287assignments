import pandas as pd 
from collections import Counter
import numpy as np
import h5py

file = open("data/train.txt")

samples_wanted = 10000000


wordcount = Counter(file.read().split())

dicti = pd.read_csv("data/words.copy.dict", sep='\t',index_col="word")

print dicti
num_words = len(dicti.index)
distribution = np.zeros(num_words)
print dicti.ix["housing","index"]

# for each word, get its index, and add it to the np vector
for item in wordcount.items(): 
	distribution[dicti.ix[item[0],"index"] -1] = item[1]

normalized = distribution/(distribution.sum())
picks = np.random.choice(num_words, size=samples_wanted, replace=True, p=normalized) + 1
print picks
with h5py.File("samples.hdf5", "w") as f:
    f['samples'] = list(picks)
