#Packages
#from email.mime import image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from tqdm import tqdm
import csv
from timeit import default_timer as timer
boolean_sample = [True, False]

BoxSize = 5
"""The pixelbox size"""

def OutputDir():
    """
    Check that the output directory exists, if not create it, also check if the .csv output file exists
    """
    outputdir = workingdir + outputname
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)
        print("Created output directory ", outputname)
        
        
        
    

def getSample(sampleinput):
    """
    returns a random sample between the minimum Boxsize and the sampleInput (height/width)
    """

    return int( ( sampleinput - BoxSize ) * np.random.random_sample())
    

def CreateTrainingSample(imageMatrix, imagenum):
    """
    Takes a matrix-converted image and returns a training sample of that image with a randomly degraded [Boxsize * Boxsize] square, and coordinates for loss function.
    
    """
    ImageHeight = imageMatrix.shape[0]
    ImageWidth = imageMatrix.shape[1]
    SampleH = getSample(ImageHeight)
    SampleW = getSample(ImageWidth)
    intSample = imageMatrix[SampleH:SampleH+BoxSize,SampleW:SampleW+BoxSize]  
    mask = np.random.choice(boolean_sample, p=[0.8, 0.2], size=(intSample.shape[:-1]))
    r = np.full((intSample.shape), 0)
    intSample[mask,:] = r[mask,:] 
    imageMatrix[SampleH:SampleH+BoxSize,SampleW:SampleW+BoxSize] = intSample
    
    # Need to add the new sample as images to the '/CompletedSamples/' folder and update the .cvs file
    convertedSample = Image.fromarray(imageMatrix)
    outputname =  str(imagenum) + ".jpg"
    convertedSample.save(sampledir + outputname)
    
    return SampleH, SampleW, BoxSize



#Main sample creation loop. 
workingdir = os.getcwd()
outputname = '/CompletedSamples/'
csvname = "Samples.csv" # this is the name of the .csv file
csvdir = workingdir + outputname + csvname
sampledir = workingdir + outputname
OutputDir() #Check if the output directory exists, if not create it. 
imagedir = workingdir + "/NewImages/" + "**/*.jpg" # NewImages is where the unaltered images must be manually placed
sortedlist = sorted(glob.glob(imagedir, recursive=True))
oldimdir = workingdir + "/NewImages/"


imagenum = 0
with open(csvdir, 'w+', newline='') as csvfile:
    for imagename in tqdm(sortedlist): # runs through the list and feeds the generator function new image samples converted to matrices, added tqdm for loading bar during runtime.
        writer = csv.writer(csvfile, delimiter=' ') # adds the write csv write handler to the file.
        imageSample = np.array(Image.open(imagename)) 
        #Rename the old .jpg file to the new number scheme
        newname = oldimdir + str(imagenum) + ".jpg"
        os.rename(imagename, newname)
        #Write the coordinates into the .csv file
        Hstart, Wstart, len = CreateTrainingSample(imageSample, imagenum)
        writer.writerow([Hstart, Wstart, len]) # H is height, not horizontal (And W is Width not W(?)ertical) :)
        imagenum += 1 

