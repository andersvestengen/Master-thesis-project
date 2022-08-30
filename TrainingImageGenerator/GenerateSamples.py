#Packages
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
channels=[0, 1, 2]
# Get names of the images in the working folder. 
workingdir = os.getcwd()
sortedlist = sorted(os.listdir(workingdir), key=len) # try and get the name of the current directory
filteredlist = [x for x in sortedlist if not ".py" in x]

# Convert images to tensors / matrices 
img = Image.open(filteredlist[0])
imageSample = np.array(img)

# Discrete channels of the image 
Channels = np.array([imageSample[:,:,i] for i in range(3)])

#print(imageSample.shape)
"""
plt.imshow(imageSample)
plt.show()
"""
# Create meshgrid for the random walk
Y, X, C = imageSample.shape
X = np.arange(X)
Y = np.arange(Y)
xx, yy = np.meshgrid(X, Y)

# Creating Random walks 
#print(xx.shape)
dim = 2
n_step = 500
step_choice = [-1, 0, 1]
origin = np.zeros((1, dim))

step_shape = np.asarray((n_step, dim))
print("this is step shape: ", step_shape)
steps = np.random.choice(a=step_choice, size=step_shape)
print("this is steps:", steps.shape)
print(steps[0])
path =  np.concatenate([origin, steps]).cumsum(0)
print(path.shape)



