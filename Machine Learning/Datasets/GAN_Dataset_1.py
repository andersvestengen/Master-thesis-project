import torch
import os
import csv
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
import glob
import numpy as np
import torch
from tqdm import tqdm
import sys
import time
"""
TODO:
    - add normalization to the image outputs [0, 255] -> [0, 1]
    - Make the pixel defect generator a part of the Pytorch Transform suite so it's dynamically created in the dataset-class.
"""
def_transform = transforms.ToTensor()
class GAN_dataset(Dataset):
    """
    TODO: 
        - update the Description
    """

    def __init__(self, training_samples=None, seed=0, BoxSize=5, workingdir=None, preprocess=False, imagefolder="/Images/", csvfolder="/CompletedSamples/", csvname="Samples.csv", transform=None):
        super(GAN_dataset, self).__init__()
        np.random.seed(seed)
        self.preprocess = preprocess
        self.BoxSize = BoxSize
        self.boolean_sample = [True, False]
        self.transform = transform
        #Setting up the directories
        self.workingdir = os.getcwd() if workingdir == None else workingdir
        self.csvdir = self.workingdir + csvfolder + csvname
        self.processedImages = self.workingdir + "/processed_images.pt"
        #self.samplecoordinates= []
        # Setting up special paths and creating the glob directory-lists here
        self.OriginalImagePathglob = self.workingdir + imagefolder + "**/*.jpg"
        
        self.OriginalImagesList = sorted(glob.glob(self.OriginalImagePathglob, recursive=True))
        if training_samples is not None:
            if len(self.OriginalImagesList) > training_samples:
                self.OriginalImagesList = self.OriginalImagesList[:training_samples]
        
        print("Number of training samples set to", len(self.OriginalImagesList))
        
        if not os.path.isfile(self.processedImages):
            print("No file detected at:")
            print(self.processedImages)
            print("Starting image processing")
            self.data = 0
            self.Preprocessor()
        else:
            print("Processed image file found, loading...")
            self.data = torch.load(self.processedImages)
        


    def getSample(self, sampleinput):
        """
        returns a random sample between the minimum Boxsize and the sampleInput (height/width)
        """

        return int( ( sampleinput - self.BoxSize ) * np.random.random_sample())

    def DefectGenerator(self, imageMatrix):
        """
        Takes a matrix-converted image and returns a training sample of that image with a randomly degraded [Boxsize * Boxsize] square, and coordinates for loss function.
        
        """
        ImageHeight = imageMatrix.shape[1]
        ImageWidth = imageMatrix.shape[2]
        SampleH = self.getSample(ImageHeight)
        SampleW = self.getSample(ImageWidth)
        intSample = imageMatrix[:,SampleH:SampleH + self.BoxSize,SampleW:SampleW + self.BoxSize] 
        mask = np.random.choice(self.boolean_sample, p=[0.8, 0.2], size=(intSample.shape[1:]))
        r = np.full((intSample.shape), 0)
        intSample[:,mask] = r[:,mask] 
        imageMatrix[:, SampleH:SampleH+self.BoxSize,SampleW:SampleW+self.BoxSize] = intSample
       
        return imageMatrix       
    
    def __len__(self):
        if self.preprocess:
            return self.data.size(0) // 2

    def Preprocessor(self):
        with tqdm(self.OriginalImagesList, unit='images') as Prepoch:
            for num, imagedir in enumerate(Prepoch):
                # Transform image and add 
                image = self.transform(Image.open(imagedir))
                sample = np.asarray(image).copy()
                sample = torch.from_numpy(self.DefectGenerator(sample))

                #stack them
                if num == 0:
                    Prepoch.set_description(f"Preprocessing images for CUDA")
                    self.data = torch.stack((image, sample), dim=0)
                else:
                    Prepoch.set_description(f"Preprocessing images for CUDA, stack size {self.data.element_size() * self.data.nelement() * 1e-6:.0f} MB")
                    self.data = torch.cat((self.data, image.unsqueeze(0)), 0)
                    self.data = torch.cat((self.data, sample.unsqueeze(0)), 0)

                    


    def __getitem__(self, index):
        if self.preprocess:
            return self.data[index*2,:], self.data[index*2+1,:] # retrieving indexes this way has been tested (in limited scope.)

  


            
        