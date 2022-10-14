import torch
import os
import csv
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
import glob
torch.manual_seed = 55
"""
TODO:
    - add normalization to the image outputs [0, 255] -> [0, 1]
    - Make the pixel defect generator a part of the Pytorch Transform suite so it's dynamically created in the dataset-class.
"""
def_transform = transforms.ToTensor()
class GAN_dataset(Dataset):
    """
    Outputs:
        The ready-made images and coordinates for loss function to train the GAN
        
    Input:
        !Within the working directory! \n
        [Image folder name, Sample folder name, csv folder name, csv filename]
    Default Input:
        [/NewImages/, /CompletedSamples/, /CompletedSamples/, /Samples.csv,]
    """
    
    def __init__(self, imagefolder="/NewImages/", samplefolder="/CompletedSamples/", csvfolder="/CompletedSamples/", csvname="Samples.csv", transform=None):
        super().__init__()
        
        self.transform = transform
        #Setting up the directories
        self.workingdir = os.getcwd()
        self.csvdir = self.workingdir + csvfolder + csvname
        self.samplecoordinates= []
        # Setting up special paths and creating the glob directory-lists here
        self.ImageSamplePathglob = self.workingdir + samplefolder + "**/*.jpg"
        self.OriginalImagePathglob = self.workingdir + imagefolder + "**/*.jpg"
        
        self.OriginalImagesList = sorted(glob.glob(self.OriginalImagePathglob, recursive=True))
        self.SampleImagesList = sorted(glob.glob(self.ImageSamplePathglob, recursive=True))
        
        #Importing the labels        
        # torch.int8() to cast into integer
        with open(self.csvdir, newline='') as csvfile:
            reader =  csv.reader(csvfile, delimiter=' ')
            self.samplecoordinates = list(reader)

    
    def __len__(self):
        # I'm (hoping) assuming the sample/image/csv lists are all the same length
        return len(self.SampleImagesList)

    def __getitem__(self, index):
        #TODO: add image augmentation
        imagedir = self.OriginalImagesList[index]
        sampledir = self.SampleImagesList[index]
        coordinates = self.samplecoordinates[index]
        
        image = torch.Tensor(Image.open(imagedir))
        sample = torch.Tensor(Image.open(sampledir))
        if self.transform is not None:
            # Can I run this twice on the image and sample, and expect the same transform to happen?
            image = self.transform(image)
            sample = self.transform(sample)
        
        return image, sample, coordinates

            
        