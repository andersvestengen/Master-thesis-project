import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import numpy as np
import torch
from tqdm import tqdm
import cv2


class GAN_dataset(Dataset):
    """
    TODO: 
        - update the Description
    """

    def __init__(self, Settings, transform=None):
        super(GAN_dataset, self).__init__()
        self.Settings = Settings
        np.random.seed(self.Settings["seed"])

        self.BoxSize = self.Settings["BoxSize"]
        self.device = self.Settings["device"]
        self.imagefolder="/Images/"
        self.name = "GAN_Dataset_1_GAN_dataset_from_folders"

        # Set incoming transform to transform
        self.transform = transform
   

        training_samples = self.Settings["Num_training_samples"]
        self.max_training_samples = training_samples
        
        #Setting up the directories
        self.workingdir = os.getcwd() if self.Settings["dataset_loc"] == None else self.Settings["dataset_loc"]
        
        #Setting up list of images
        self.OriginalImagePathglob = self.workingdir + self.imagefolder + "**/*.jpg"
        self.OriginalImagesList = sorted(glob.glob(self.OriginalImagePathglob, recursive=True))
                
        if training_samples is None:
            print("training samples is none")
            training_samples = len(self.OriginalImagesList)
        
        print("Number of training samples set to", len(self.OriginalImagesList))

        if len(self.OriginalImagesList) == 0:
            raise Exception(f"Found no local training images at {self.workingdir + self.imagefolder} !")            
        else:
            if training_samples != len(self.OriginalImagesList):
                print("number of images does not equal input parameter [",(len(self.OriginalImagesList)),"/",training_samples,"] adjusting")
                if len(self.OriginalImagesList) > training_samples:
                    self.OriginalImagesList = self.OriginalImagesList[:training_samples]
                else:
                    print(f"Number of desired training images is bigger than number of available images \n {len(self.OriginalImagesList)} < {training_samples}")

            print("Number of images is:", len(self.OriginalImagesList))



    def getSample(self, sampleinput):
        """
        returns a random sample between the minimum Boxsize and the sampleInput (height/width)
        """

        return int( ( sampleinput - self.BoxSize ) * np.random.random_sample())

    def DefectGenerator(self, imageMatrix):
        """
        Takes a matrix-converted image and returns a training sample of that image with a randomly degraded [Boxsize * Boxsize] square, and coordinates for loss function.
        
        """
        ImageHeight = imageMatrix.size(1)
        ImageWidth = imageMatrix.size(2)
        SampleH = self.getSample(ImageHeight)
        SampleW = self.getSample(ImageWidth)
        intSample = imageMatrix[:,SampleH:SampleH + self.BoxSize,SampleW:SampleW + self.BoxSize] 
        mask = torch.randint(0,2, (intSample.size()[1:])).bool()
        r = torch.full((intSample.size()), 0).float()
        intSample[:,mask] = r[:,mask] 
        imageMatrix[:,SampleH:SampleH+self.BoxSize,SampleW:SampleW+self.BoxSize] = intSample
       
        return imageMatrix, [SampleH, SampleW, self.BoxSize]
    
    def __len__(self):
            return len(self.OriginalImagesList)

    def resize_im(self, image):
        sizes = [256, 256]
        return cv2.resize(image, sizes, interpolation= cv2.INTER_LINEAR)

    def CenterCrop(self, image, val=256):
        center = image.shape
        if center[0] < 256 or center[1] < 256:
            image = self.resize_im(image)
            center = image.shape

        x = np.around((center[1]*0.5 - val*0.5), 0).astype(np.int8)
        y = np.around((center[0]*0.5 - val*0.5),0).astype(np.int8)
        return image[y:y+val, x:x+val]


    def load_image(self, path):
        image = self.CenterCrop(cv2.imread(str(path)))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255

    def totensortorch(self, image):
        return torch.from_numpy(np.moveaxis(image, -1, 0)).float()


    def __getitem__(self, idx):
        if self.transform is not None:
            target = self.transform(self.totensortorch(self.load_image(self.OriginalImagesList[idx])))
        else:
            target = self.totensortorch(self.load_image(self.OriginalImagesList[idx]))

        defect, arr = self.DefectGenerator(target.clone())
        
        arr = torch.tensor(arr)

        return target, defect, arr




  


            
        