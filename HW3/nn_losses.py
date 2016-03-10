#import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

res = np.loadtxt("nn_losses.txt", skiprows=1, delimiter=",")
print res
plt.figure()
plt.plot(res[:, 0], res[:, 1])
plt.title("NNLM Loss Diagram")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()