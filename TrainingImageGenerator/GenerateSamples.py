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

