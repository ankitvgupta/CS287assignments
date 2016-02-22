# Usage python loss_diagram.py
import pandas as pd 
import sys
import seaborn as sns
import matplotlib.pyplot as plt 

sns.set_context("paper")

data_lr = pd.read_csv("losses_lr.txt")
data_nn = pd.read_csv("losses_nn.txt")
data_nnpre = pd.read_csv("losses_nnpre.txt")

plt.figure()
plt.title("Logistic Regression Loss")
plt.plot(data_lr.values[:,0], data_lr.values[:,1])
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.figure()
plt.title("Neural Network Loss")
plt.plot(data_nn.values[:,0], data_nn.values[:,1])
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.figure()
plt.title("Neural Network (Pretrained Embeddings) Loss")
plt.plot(data_nnpre.values[:,0], data_nnpre.values[:,1])
plt.xlabel("Epochs")
plt.ylabel("Loss")


plt.show()
