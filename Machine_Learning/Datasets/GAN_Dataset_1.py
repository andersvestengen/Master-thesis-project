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

def_transform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor()
])

class GAN_dataset(Dataset):
    """
    TODO: 
        - update the Description
    """

    def __init__(self, preprocess_storage=None, training_samples=None, seed=0, BoxSize=5, workingdir=None, imagefolder="/Images/", transform=None, device="cpu"):
        super(GAN_dataset, self).__init__()
        np.random.seed(seed)
        self.training_process_name = "/processed_images.pt"
        self.preprocess_storage = preprocess_storage
        self.BoxSize = BoxSize
        self.device = device
        self.boolean_sample = [True, False]
        if transform is not None:
            self.transform = transform
        else:
            self.transform = def_transform          
        self.max_training_samples = training_samples
        #Setting up the directories
        self.workingdir = os.getcwd() if workingdir == None else workingdir
        self.Small_cache_storage = self.workingdir + self.training_process_name
        #self.samplecoordinates= []
        # Setting up special paths and creating the glob directory-lists here
        self.OriginalImagePathglob = self.workingdir + imagefolder + "**/*.jpg"
        
        self.OriginalImagesList = sorted(glob.glob(self.OriginalImagePathglob, recursive=True))
        if training_samples is not None and  (len(self.OriginalImagesList) > training_samples):
            if len(self.OriginalImagesList) > training_samples:
                self.OriginalImagesList = self.OriginalImagesList[:training_samples]
        
        print("Number of training samples set to", len(self.OriginalImagesList))
        
        #replace storage string if no serverside storage is set
        if preprocess_storage:
            self.Large_cache_storage = preprocess_storage + self.training_process_name  
        else:
            self.preprocess_storage = self.workingdir
            self.Large_cache_storage = self.Small_cache_storage
        
        if not os.path.isfile(self.Large_cache_storage):
            print("No file detected at:")
            print(self.Large_cache_storage)
            print("Starting image processing")
            self.data = 0
            self.Preprocessor()
        else:
            print("Processed image file found, loading...")
            start = time.time()
            self.data = torch.load(self.Large_cache_storage)
            stop = time.time()
            print(f"Time spent loading file was: {stop - start:.2} seconds")
            print("Number of images was:", self.data.size(0) // 2)
            
        


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
            return self.data.size(0) // 2

    def Preprocessor(self):
        with tqdm(self.OriginalImagesList, unit='images') as Prepoch:
            for num, imagedir in enumerate(Prepoch):
                # Transform image and add 
                image = self.transform(Image.open(imagedir))
                sample = np.asarray(image).copy()
                sample = torch.from_numpy(self.DefectGenerator(sample))
                # If imagelist > 200 and num > 200
                if (num > 0) and (num % 200 == 0):
                    torch.save(self.data, self.preprocess_storage+"/processed_images"+str(num)+".pt")
                    self.data = 0                  

                if num == 0 or num % 200 == 0:
                    Prepoch.set_description(f"Preprocessing images for CUDA")
                    self.data = torch.stack((image, sample), dim=0)
                else:
                    Prepoch.set_description(f"Preprocessing images for CUDA, stack size {self.data.element_size() * self.data.nelement() * 1e-6:.0f} MB")
                    self.data = torch.cat((self.data, image.unsqueeze(0)), 0)
                    self.data = torch.cat((self.data, sample.unsqueeze(0)), 0)
            if not isinstance(self.data, int):
                print("trying to save incomplete last cache size")
                torch.save(self.data, self.preprocess_storage+"/processed_images_last.pt")
        print("reconstituting images into single file:")
        self.data = 0
        cache_list = sorted(glob.glob(self.preprocess_storage + "**/*.pt", recursive=True))
        with tqdm(cache_list, unit='patches') as Crepoch:
            for num, cache in enumerate(Crepoch):

                if num == 0:
                    self.data = torch.load(cache)
                    os.remove(cache)
                else:
                    Crepoch.set_description(f"Stacking cache for CUDA, stack size {self.data.element_size() * self.data.nelement() * 1e-6:.0f} MB")
                    temp = torch.load(cache)
                    self.data = torch.cat((self.data, temp), 0)
                    os.remove(cache)
        print("Saving to file")
        start = time.time()
        torch.save(self.data, self.Large_cache_storage)
        stop = time.time()
        print(f"Done, time taken was: {stop - start:.0f} seconds")

                    


    def __getitem__(self, index):
            return self.data[index*2,:], self.data[index*2+1,:] # retrieving indexes this way has been tested (in limited scope.)

  


            
        