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

    def __init__(self, Settings, transform=None):
        super(GAN_dataset, self).__init__()
        self.Settings = Settings
        np.random.seed(self.Settings["seed"])
        self.training_process_name = "/processed_images.pt"
        self.preprocess_storage = self.Settings["preprocess_storage"]
        self.BoxSize = self.Settings["BoxSize"]
        self.device = self.Settings["Datahost"]
        self.imagefolder="/Images/"
        self.boolean_sample = [True, False]
        if transform is not None:
            self.transform = transform
        else:
            self.transform = def_transform          
        training_samples = self.Settings["Num_training_samples"]
        self.max_training_samples = training_samples
        
        #Setting up the directories
        self.workingdir = os.getcwd() if self.Settings["dataset_loc"] == None else self.Settings["dataset_loc"]
        self.Small_cache_storage = self.workingdir + self.training_process_name
        
        # Setting up special paths and creating the glob directory-lists here
        self.OriginalImagePathglob = self.workingdir + self.imagefolder + "**/*.jpg"
        
        self.OriginalImagesList = sorted(glob.glob(self.OriginalImagePathglob, recursive=True))
         
        if training_samples is not None and  (len(self.OriginalImagesList) > training_samples):
            if len(self.OriginalImagesList) > training_samples:
                self.OriginalImagesList = self.OriginalImagesList[:training_samples]
                
        if training_samples is None:
            print("training samples is none")
            training_samples = len(self.OriginalImagesList)
        
        print("Number of training samples set to", len(self.OriginalImagesList))
        
        #replace storage string if no serverside storage is set
        if self.preprocess_storage:
            self.Large_cache_storage = self.preprocess_storage + self.training_process_name  
        else:
            self.preprocess_storage = self.workingdir
            self.Large_cache_storage = self.Small_cache_storage
        
        if not os.path.isfile(self.Large_cache_storage):
            if len(self.OriginalImagesList) == 0:
                raise Exception(f"Found no local training images at {self.workingdir + self.imagefolder} ! \n And no preprocess file at {self.Large_cache_storage} !")            
            print("No file detected at:")
            print(self.Large_cache_storage)
            print("Starting image processing")
            self.data = 0
            self.Preprocessor()
        else:
            print("Processed image file found, loading...")
            start = time.time()
            self.data = torch.load(self.Large_cache_storage, map_location=torch.device(self.device))
            stop = time.time()
            print(f"Time spent loading file was: {stop - start:.2} seconds")
            if training_samples != self.data.size(0) // 2:
                print("images in cache_size does not equal input parameter [",(self.data.size(0) // 2),"/",training_samples,"] adjusting")
                if training_samples < self.data.size(0) // 2:
                    newsize = self.data.size(0) // 2 - training_samples
                    self.data = self.data[:-newsize*2]
                else:
                    self.data = 0
                    self.Preprocessor()
            print("Number of images is:", self.data.size(0) // 2)
            
        


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
        mask = np.random.choice(self.boolean_sample, p=[1, 0], size=(intSample.shape[1:]))
        #mask = np.ones(intSample.shape[1:])
        r = np.full((intSample.shape), 0)
        intSample[:,mask] = r[:,mask] 
        imageMatrix[:, SampleH:SampleH+self.BoxSize,SampleW:SampleW+self.BoxSize] = intSample
        #defect_region = np.asarray([SampleH, SampleW, self.BoxSize])
       
        return imageMatrix#, defect_region       
    
    def __len__(self):
            return self.data.size(0) // 2

    def Preprocessor(self):
        with tqdm(self.OriginalImagesList, unit='images') as Prepoch:
            for num, imagedir in enumerate(Prepoch):
                # Transform image and add 
                image = self.transform(Image.open(imagedir))
                sample = np.asarray(image).copy()
                sample = torch.from_numpy(self.DefectGenerator(sample))

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
        if self.Settings["Datahost"] == "cuda": # Move to GPU if available
            print("Loading to device")
            self.data.to(self.device)

                    


    def __getitem__(self, index):
            return self.data[index*2,:], self.data[index*2+1,:] # retrieving indexes this way has been tested (in limited scope.)

  


            
        