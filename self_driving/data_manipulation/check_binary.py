import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

command2int = {"up": 0, "down": 1, "left": 2, "right": 3}
int2command = {i[1]:i[0] for i in command2int.items()}

data = np.load("data.npy")
labels = np.load("labels.npy")

print("data shape = {}".format(data.shape))
print("data type = {}".format(data.dtype))
print("labels shape = {}".format(labels.shape))
print("labels type = {}".format(labels.dtype))

n = 35800
# n = 1995 


'''
RGB image test
'''
# example = data[n].reshape((90, 160,3))
# img = Image.fromarray(example, 'RGB')
# img.show()



'''
binary image test
'''
example = data[n].reshape((90, 160))
img = Image.fromarray(example, 'L')
img.show()



'''
green channel image test
'''
# example = data[n].reshape((90, 160))
# plt.imshow(example,cmap=plt.cm.Greens_r)
# plt.show()

print("\naction = {}".format(int2command[labels[n][0]]))