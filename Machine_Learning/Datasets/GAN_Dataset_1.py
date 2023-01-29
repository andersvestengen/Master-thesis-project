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

    def __init__(self, Settings, transform=None, preprocess=False):
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

        #Create preprocess storage if condition is met 
        self.preprocess = preprocess
        self.preprocess_storage = self.Settings["preprocess_storage"]
        self.preprocess_cache = self.preprocess_storage + "/processed_images.pt"
        if self.preprocess and not os.isdir(self.preprocess_storage):
            os.makedirs(self.preprocess_storage)
            self.ImagePreprocessor()
        elif self.preprocess and not os.exists(self.preprocess_cache):
            self.ImagePreprocessor()
        elif self.preprocess and os.exists(self.preprocess_cache):
            self.data = torch.load(self.preprocess_cache, map_location="cpu")
        
        if self.data.size(0) != len(self.OriginalImagesList):
            print("Number of cached images not equal to the amount of images selected[", self.data.size(0), " | ", len(self.OriginalImagesList),"]" )
            self.data = 0
            self.ImagePreprocessor()
            
    def getSample(self, Total_length):
        """
        returns a random sample between the minimum Boxsize and the Total_length (height/width)
        """
        margin = self.BoxSize * self.Settings["Loss_region_Box_mult"]
        sample = int( (Total_length - margin) * np.random.random_sample() )

        if sample < margin: # so there's both lower and upper margin
            return int(margin)
        else:
            return sample

    def ImagePreprocessor(self):
        with tqdm(self.OriginalImagesList, unit='images') as Prepoch:
            for num, imagedir in enumerate(Prepoch):
                # Transform image and add 
                image = self.transform(Image.open(imagedir))

                if (num > 0) and (num % 200 == 0):
                    torch.save(self.data, self.preprocess_storage+"/processed_images"+str(num)+".pt")
                    self.data = 0                  

                if num == 0 or num % 200 == 0:
                    Prepoch.set_description(f"Preprocessing images for CUDA")
                    self.data = image
                else:
                    Prepoch.set_description(f"Preprocessing images for CUDA, stack size {self.data.element_size() * self.data.nelement() * 1e-6:.0f} MB")
                    self.data = torch.cat((self.data, image.unsqueeze(0)), 0)
            if not isinstance(self.data, int):
                print("trying to save incomplete last cache size")
                torch.save(self.data, self.preprocess_storage+"/processed_images_last.pt")
        print("reconstituting images into single file:")
        self.data = 0
        cache_list = sorted(glob.glob(self.preprocess_storage + "**/*.pt", recursive=True))
        with tqdm(cache_list, unit='patches', leave=True) as Crepoch:
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
        torch.save(self.data, self.preprocess_cache)



    def DefectGenerator(self, imageMatrix):
        """
        Takes a matrix-converted image and returns a training sample of that image with a randomly degraded [Boxsize * Boxsize] square, and coordinates for loss function.
        
        """
        ImageY = imageMatrix.size(1)
        ImageX = imageMatrix.size(2)
        SampleY = self.getSample(ImageY)
        SampleX = self.getSample(ImageX)
        intSample = imageMatrix[:,SampleY:SampleY + self.BoxSize, SampleX:SampleX + self.BoxSize] 
        mask = torch.randint(0,2, (intSample.size()[1:])).bool()
        r = torch.full((intSample.size()), 0).float()
        intSample[:,mask] = r[:,mask] 
        imageMatrix[:,SampleY:SampleY+self.BoxSize,SampleX:SampleX+self.BoxSize] = intSample

        return imageMatrix, [SampleY, SampleX, self.BoxSize]
    
    def __len__(self):
        if self.preprocess:
            return self.data.size(0)
        else:
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
        #imread returns Y,X,C
        image = self.CenterCrop(cv2.imread(str(path)))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255

    def totensortorch(self, image):
        return torch.from_numpy(np.moveaxis(image, -1, 0)).float()


    def __getitem__(self, idx):
        if self.transform is not None:
            if self.preprocess:
                target = self.transform(self.data[idx,:])
            else:
                target = self.transform(self.totensortorch(self.load_image(self.OriginalImagesList[idx])))
        else:
            if self.preprocess:
                target = self.data[idx,:]
            else:
                target = self.totensortorch(self.load_image(self.OriginalImagesList[idx]))

        defect, arr = self.DefectGenerator(target.clone())
        
        arr = torch.tensor(arr)

        return target, defect, arr




  


            
        