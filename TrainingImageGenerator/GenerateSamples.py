#Packages
from email.mime import image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from timeit import default_timer as timer
boolean_sample = [True, False]
rnum = np.random.random_sample() 
""" numpy random_sample alias"""

BoxSize = 5
"""The pixelbox size"""

def getSample(sampleinput):
    """
    returns a random sample between the minimum Boxsize and the sampleInput (height/width)
    """
    return int( ( sampleinput - BoxSize ) * rnum)

def CreateTrainingSample(imageMatrix):
    """
    Takes a matrix-converted image and returns a training sample of that image
    
    """
    ImageHeight = imageMatrix.shape[0]
    ImageWidth = imageMatrix.shape[1]
    SampleH = getSample(ImageHeight)
    SampleW = getSample(ImageWidth)
    intSample = imageMatrix[SampleH:SampleH+BoxSize,SampleW:SampleW+BoxSize,:]  
    mask = np.random.choice(boolean_sample, p=[0.8, 0.2], size=(intSample.shape[:-1]))
    r = np.full((intSample.shape), 0)
    intSample[mask,:] = r[mask,:] 
    imageMatrix[SampleH:SampleH+BoxSize,SampleW:SampleW+BoxSize,:] = intSample
    
    return imageMatrix

"""
Dev notes:
    - Algorithm to be simplified
    - No more random walks etc.
    - Algorithm will:
        - sample a random 10x10 pixel box of the image
        - remove a random subset of that sample
        - implant the randomly constructed 'defect'

"""


# Get names of the images in the working folder. 
workingdir = os.getcwd()
sortedlist = sorted(os.listdir(workingdir), key=len) # try and get the name of the current directory
print(sortedlist)

filteredlist = [x for x in sortedlist if ".jpg" in x]
print(filteredlist)

img = Image.open(filteredlist[0])
imageSample = np.array(img)


#start = timer()
TimeSampler = CreateTrainingSample(imageSample)
#stop = timer()
#print("elapsed time for creating training sample was:", (stop - start)*1E3, " ms")
convertedSample = Image.fromarray(TimeSampler)
convertedSample.save("BasicTrainingSample.jpg")
