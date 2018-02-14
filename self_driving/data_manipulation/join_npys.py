import numpy as np

oned = np.load("data1.npy")
onel = np.load("labels1.npy")
twod = np.load("data2.npy")
twol = np.load("labels2.npy")

data = np.concatenate((oned,twod), axis=0) 
labels = np.concatenate((onel,twol), axis=0)

np.save("data.npy", data)
np.save("labels.npy", labels)