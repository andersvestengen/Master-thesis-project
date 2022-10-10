import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
import glob

class GAN_dataset(Dataset):
    def __init__(self, dir=None, transform=None):
        self.dir = dir
        super().__init__()
        self.workingdir = os.getcwd()
        self.csvdir = self.workingdir + "/Samples.csv"
        self.ImageSamplePath = 
        self.OriginalImagePathglob = self.workingdir + "/NewImages/" + "**/*.jpg"
        # torch.int8() to cast into integer
        #Importing the labels
        with open('')

    
    def __len__(self):
        # TODO: return number of imagesamples.
        pass

    def __getitem__(self, index):
        #TDOD: make function return i'th sample
        return super().__getitem__(index)