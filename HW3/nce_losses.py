#import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

res = np.loadtxt("nce_losses.txt", skiprows=1, delimiter=",")
xs = np.array(range(len(res)))

plt.figure()
plt.plot(xs[::300], res[::300])
plt.title("NCE Loss Diagram")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()