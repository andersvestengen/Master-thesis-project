import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import numpy as np
import torch
from tqdm import tqdm
from math import ceil

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
        #self.training_process_name = "/processed_images.pt"
        self.preprocess_storage = self.Settings["preprocess_storage"]
        self.BoxSize = self.Settings["BoxSize"]
        self.device = self.Settings["Datahost"]
        self.imagefolder="/Images/"
        self.totensor = transforms.ToTensor()

        #Create folders for outputs
        if self.Settings["preprocess_storage"] is None:
            if not os.path.exists(self.Settings["dataset_loc"] + "/Processed_Images"):
                print("Couldn't find processed image folder!")
                os.makedirs(self.Settings["dataset_loc"] + "/Processed_Images/")
                os.makedirs(self.Settings["dataset_loc"] + "/Processed_Images/Targets/")
                os.makedirs(self.Settings["dataset_loc"] + "/Processed_Images/Defects/")

            self.OutputFolder = self.Settings["dataset_loc"] + "/Processed_Images"
            self.OutputFolderTargets = self.OutputFolder + "/Targets"
            self.OutputFolderDefects = self.OutputFolder + "/Defects"
        
        else:
            if not os.path.exists(self.Settings["preprocess_storage"] + "/Processed_Images"):
                os.makedirs(self.Settings["preprocess_storage"] + "/Processed_Images")
                os.makedirs(self.Settings["preprocess_storage"] + "/Processed_Images/Targets")
                os.makedirs(self.Settings["preprocess_storage"] + "/Processed_Images/Defects")

            self.OutputFolder = self.Settings["preprocess_storage"] + "/Processed_Images"
            self.OutputFolderTargets =  self.OutputFolder + "/Targets"
            self.OutputFolderDefects =  self.OutputFolder + "/Defects"            

        self.boolean_sample = [True, False]
        if transform is not None:
            self.transform = transform
        else:
            self.transform = def_transform    

        training_samples = self.Settings["Num_training_samples"]
        self.max_training_samples = training_samples
        
        #Setting up the directories
        self.workingdir = os.getcwd() if self.Settings["dataset_loc"] == None else self.Settings["dataset_loc"]
        #self.Small_cache_storage = self.workingdir + self.training_process_name
        
        # Setting up special paths and creating the glob directory-lists here
        self.OriginalImagePathglob = self.workingdir + self.imagefolder + "**/*.jpg"
        
        self.OriginalImagesList = sorted(glob.glob(self.OriginalImagePathglob, recursive=True))
         
        if training_samples is not None and  (len(self.OriginalImagesList) > training_samples):
            self.OriginalImagesList = self.OriginalImagesList[:training_samples]
                
        if training_samples is None:
            print("training samples is none")
            training_samples = len(self.OriginalImagesList)
        
        print("Number of training samples set to", len(self.OriginalImagesList))
        
        #replace storage string if no serverside storage is set
        """        
        if self.preprocess_storage:
            self.Large_cache_storage = self.preprocess_storage + self.training_process_name  
        else:
            self.preprocess_storage = self.workingdir
            self.Large_cache_storage = self.Small_cache_storage
        """
        self.targetglob = self.OutputFolderTargets + "**/*.jpg"
        self.targetlist = sorted(glob.glob(self.targetglob, recursive=True))

        self.defectglob = self.OutputFolderDefects + "**/*.jpg"
        self.defectlist = sorted(glob.glob(self.defectglob, recursive=True))

        if len(self.targetlist) == 0:
            if len(self.OriginalImagesList) == 0:
                raise Exception(f"Found no local training images at {self.workingdir + self.imagefolder} ! \n And no preprocess files at {self.OutputFolder} !")            
            print("No files detected at:")
            print(self.OutputFolder)
            print("Starting image processing")
            self.Preprocessor()
        else:
            print("Processed image file found, loading...")
            if training_samples != len(self.targetlist):
                print("images in cache_size does not equal input parameter [",(len(self.targetlist)),"/",training_samples,"] adjusting")
                if training_samples < len(self.targetlist):
                    newsize = len(self.targetlist) - training_samples
                    self.data = self.data[:-newsize]
                else:
                    self.Preprocessor()
            print("Number of images is:", len(self.targetlist))
            
        


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
        mask = np.random.choice(self.boolean_sample, p=[0.5, 0.5], size=(intSample.shape[1:]))
        #mask = np.ones(intSample.shape[1:])
        r = np.full((intSample.shape), 0)
        intSample[:,mask] = r[:,mask] 
        imageMatrix[:, SampleH:SampleH+self.BoxSize,SampleW:SampleW+self.BoxSize] = intSample
        #defect_region = np.asarray([SampleH, SampleW, self.BoxSize])
       
        return imageMatrix#, defect_region       
    
    def __len__(self):
            return len(self.targetlist)

    def CenterCrop(self, image, value): # Only for PIL images
        width, height = image.size

        left = ceil((width - value)/2)
        top = ceil((height - value)/2)
        right = ceil((width + value)/2)
        bottom = ceil((height + value)/2)

        return image.crop((left, top, right, bottom))


    def Preprocessor(self):
        with tqdm(self.OriginalImagesList, unit='images') as Prepoch:
            for num, imagedir in enumerate(Prepoch):
                # Transform image and add 
                Prepoch.set_description(f"Preprocessing images for Model training")
                target = Image.open(imagedir)
                target = self.CenterCrop(target, 256)
                target.show()
                target.save(self.OutputFolderTargets + "/" + str(num) + ".jpg", "JPEG")
                defect = Image.fromarray(self.DefectGenerator(np.asarray(target).copy()))     
                defect.show()
                defect.save(self.OutputFolderDefects + "/" + str(num) + ".jpg", "JPEG")
                break

        self.targetglob = self.OutputFolderTargets + "**/*.jpg"
        self.targetlist = sorted(glob.glob(self.targetglob, recursive=True))

        self.defectglob = self.OutputFolderDefects + "**/*.jpg"
        self.defectlist = sorted(glob.glob(self.defectglob, recursive=True))

    def __getitem__(self, index):
        #Add transform here    
        target = self.totensor(Image.open(self.targetlist[index]))
        defect = self.totensor(Image.open(self.defectlist[index]))


        cat_transform = torch.cat((target.unsqueeze(0), defect.unsqueeze(0)),0)
        
        # Apply the transformations to both images simultaneously:
        transformed_images = self.transform(cat_transform)

        return transformed_images[0].squeeze(0), transformed_images[1].squeeze(0)

  


            
        