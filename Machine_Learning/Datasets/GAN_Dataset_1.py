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

        self.preprocess_storage = self.Settings["preprocess_storage"]
        self.BoxSize = self.Settings["BoxSize"]
        self.device = self.Settings["Datahost"]
        self.imagefolder="/Images/"
        self.totensor = transforms.ToTensor()
        self.name = "GAN_Dataset_1_GAN_dataset_from_folders"

        self.totensorcrop = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.CenterCrop(self.Settings["ImageHW"]),
                        ])

        self.toPIL = transforms.ToPILImage()

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
                    self.targetlist = self.targetlist[:-newsize]
                    self.defectlist = self.defectlist[:-newsize]
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
        ImageHeight = imageMatrix.shape[0]
        ImageWidth = imageMatrix.shape[1]
        SampleH = self.getSample(ImageHeight)
        SampleW = self.getSample(ImageWidth)
        intSample = imageMatrix[SampleH:SampleH + self.BoxSize,SampleW:SampleW + self.BoxSize,:] 
        mask = np.random.choice(self.boolean_sample, p=[0.7, 0.3], size=(intSample.shape[:-1]))
        r = np.full((intSample.shape), 0)
        intSample[mask,:] = r[mask,:] 
        imageMatrix[SampleH:SampleH+self.BoxSize,SampleW:SampleW+self.BoxSize,:] = intSample
       
        return imageMatrix    
    
    def __len__(self):
            return len(self.targetlist)

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

    def Preprocessor(self):
        with tqdm(self.OriginalImagesList, unit='images') as Prepoch:
            for num, imagedir in enumerate(Prepoch):
                Prepoch.set_description(f"Preprocessing images for Model training")
                target = self.CenterCrop(cv2.imread(imagedir))
                target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
                defect = Image.fromarray(self.DefectGenerator(target.copy()))
                target = Image.fromarray(target)
                target.save(self.OutputFolderTargets + "/" + str(num) + ".jpg", "JPEG")
                defect.save(self.OutputFolderDefects + "/" + str(num) + ".jpg", "JPEG")
        self.targetglob = self.OutputFolderTargets + "**/*.jpg"
        self.targetlist = sorted(glob.glob(self.targetglob, recursive=True))

        self.defectglob = self.OutputFolderDefects + "**/*.jpg"
        self.defectlist = sorted(glob.glob(self.defectglob, recursive=True))

    def load_image(self, path):
        image = cv2.imread(str(path))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255

    def totensortorch(self, image):
        return torch.from_numpy(np.moveaxis(image, -1, 0)).float()

    def __getitem__(self, idx):
        target = self.totensortorch(self.load_image(self.targetlist[idx]))
        defect = self.totensortorch(self.load_image(self.defectlist[idx]))
        cat_transform = torch.cat((target.unsqueeze(0), defect.unsqueeze(0)), 0)
        transformed_images = self.transform(cat_transform)
        

        return transformed_images[0].squeeze(0), transformed_images[1].squeeze(0)




  


            
        