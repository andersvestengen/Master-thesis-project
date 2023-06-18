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

    def __init__(self, Settings, transform=None, preprocess=True):
        super(GAN_dataset, self).__init__()
        self.Settings = Settings
        if not self.Settings["seed"] == None:
            self.rng = np.random.default_rng(self.Settings["seed"])
            self.defect_seed = torch.Generator()
            self.defect_seed.manual_seed(self.Settings["seed"])
        else:
            self.rng = np.random.default_rng()
            self.defect_seed = torch.Generator()
            self.defect_seed.manual_seed()
        self.BoxSet = self.Settings["BoxSet"]
        self.device = self.Settings["device"]
        self.Blockmode = self.Settings["Blockmode"]
        self.CenterDefect = self.Settings["CenterDefect"]
        if Settings["Defect_mode"] == "Inference":
            self.InferenceMode = True
        else:
            self.InferenceMode = False
        #Define wether defects are blacked out, whited out or both
        self.BlackWhite = torch.tensor(self.Settings["BlackorWhite"], dtype=torch.int8)
        if self.Settings["BlackorWhite"][0] == True:
            self.BlackWhite[0] = 0
        else:
            self.BlackWhite[0] = 1

        if self.Settings["BlackorWhite"][1] == True:
            # values are +1, due to torch.randint 'high' being exclusive, while low is inclusive
            self.BlackWhite[1] = 2
        else:
            self.BlackWhite[1] = 1

        self.mean = self.Settings["Data_mean"]
        self.std = self.Settings["Data_std"]
        self.imagefolder = self.Settings["Dataset_loc"]

        if preprocess:
            self.name = "GAN_Dataset_1_GAN_dataset_caching"
        else:
            self.name = "GAN_Dataset_1_GAN_dataset_from_folders"

        # Set conversion transform during preproccessing
        self.ConvertToTensor = transforms.ToTensor()        

        if self.Settings["Do norm"]:
            self.Image_To_Sample_Transform = torch.nn.Sequential(
                    # Constants calculated using the Dataset_Check_Norm.py script
                    transforms.Normalize(mean=self.mean,
                                        std=self.std),
                    transforms.CenterCrop(self.Settings["ImageHW"]),
            )
        else:
            print("Dataset: Normalization is off")
            self.Image_To_Sample_Transform = transforms.CenterCrop(self.Settings["ImageHW"])


        # Set incoming transform to transform
        self.transform = transform
   

        training_samples = self.Settings["Num_training_samples"]
        self.max_training_samples = training_samples
        
        #Setting up the directories
        self.workingdir = os.getcwd() if self.Settings["dataset_loc"] == None else self.Settings["dataset_loc"]
        
        #Setting up list of images
        self.OriginalImagePathglob = self.imagefolder + "**/*.jpg"
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

        #Legacy conditions (delete?)
        #Create preprocess storage if condition is met 
        self.preprocess = preprocess
        if self.preprocess:
            self.preprocess_storage = self.Settings["preprocess_storage"]

            if self.preprocess_storage is None:
                self.preprocess_storage = self.workingdir + "/Processing_storage"

            self.preprocess_cache = self.preprocess_storage + "/processed_images.pt"

            if not os.path.isdir(self.preprocess_storage):
                os.makedirs(self.preprocess_storage)
                self.ImagePreprocessor()

            if not os.path.exists(self.preprocess_cache):
                self.ImagePreprocessor()
            else:
                print("Loading cache data from file")
                self.data = torch.load(self.preprocess_cache, map_location="cpu")
        
            if self.data.size(0) != len(self.OriginalImagesList):
                print("Number of cached images not equal to the amount of images selected[", self.data.size(0), " | ", len(self.OriginalImagesList),"]" )
                if self.data.size(0) > len(self.OriginalImagesList):
                    self.data = self.data[:len(self.OriginalImagesList)]
                else:
                    self.data = 0
                    os.remove(self.preprocess_cache)
                    self.ImagePreprocessor()

    #Legacy (delete?)
    def ImagePreprocessor(self):
        with tqdm(self.OriginalImagesList, unit='images') as Prepoch:
            for num, imagedir in enumerate(Prepoch):
                # Transform image and add 
                image = self.Image_To_Sample_Transform(self.load_torch_image(imagedir))

                if (num > 0) and (num % 200 == 0):
                    torch.save(self.data, self.preprocess_storage+"/processed_images"+str(num)+".pt")
                    self.data = 0                  

                if num == 0 or num % 200 == 0:
                    Prepoch.set_description(f"Preprocessing images for CUDA")
                    self.data = image.unsqueeze(0)
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


    def getSample(self, Total_length, BoxSize):
        """
        returns a random sample between the minimum Boxsize and the Total_length (height/width)
        """
        margin = torch.tensor(BoxSize * self.Settings["Loss_region_Box_mult"]).to(torch.uint8)
        sample = ((Total_length - margin) * torch.rand(1)).to(torch.uint8).clamp(margin)

        return sample

    def DefectGenerator(self, imageMatrix):
        """
        Takes a matrix-converted image and returns a training sample of that image with a randomly degraded [Boxsize * Boxsize] square, and coordinates for loss function.
        
        """
        BoxSize = torch.randint(self.BoxSet[0],self.BoxSet[1] + 1, (1,)).to(torch.uint8)
        ImageY = imageMatrix.size(1) # Height
        ImageX = imageMatrix.size(2) # Width
        if self.CenterDefect:
            SampleY = int(ImageY // 2)
            SampleX = int(ImageX // 2)
        else:
            SampleY = self.getSample(ImageY, self.BoxSet[1])
            SampleX = self.getSample(ImageX, self.BoxSet[1])

        #color = torch.randint(self.BlackWhite[0],self.BlackWhite[1], (1,), generator=self.defect_seed)
        #Doing black for now
        if self.Blockmode:
            #Create a solid block in the image, used in eatly training
            minimask = torch.full((BoxSize, BoxSize), 0).float()
            imageMatrix[:,SampleY:SampleY + BoxSize, SampleX:SampleX + BoxSize] = minimask
            Mask = torch.ones(imageMatrix.size())
            Mask[:,SampleY:SampleY + BoxSize, SampleX:SampleX + BoxSize] = torch.zeros((BoxSize, BoxSize))
        else:
            #Create a more complex defect in the image
            defect_mask = torch.randint(0,2, ((BoxSize, BoxSize))).bool()
            Cutout = imageMatrix[:,SampleY:SampleY + BoxSize, SampleX:SampleX + BoxSize]
            r = torch.full(((3, BoxSize, BoxSize)), 0).float()[:,defect_mask]
            Cutout[:,defect_mask] = r
            imageMatrix[:,SampleY:SampleY + BoxSize, SampleX:SampleX + BoxSize] = Cutout
            #Create Mask
            Mask = torch.ones(imageMatrix.size())
            MCutout = torch.ones((3, BoxSize, BoxSize))
            MCutout[:,defect_mask] = r
            Mask[:,SampleY:SampleY + BoxSize, SampleX:SampleX + BoxSize] = MCutout

        if self.InferenceMode:
            return imageMatrix, [SampleY, SampleX, BoxSize.item()]
        else:
            return imageMatrix, Mask
    

    def __len__(self):
        if self.preprocess:
            return self.data.size(0)
        else:
            return len(self.OriginalImagesList)

    #Legacy (delete?)
    def resize_im(self, image): # REDUNDANT
        sizes = [256, 256]
        return cv2.resize(image, sizes, interpolation= cv2.INTER_LINEAR)

    #Legacy (delete?)
    def CenterCrop(self, image, val=256): # Redundant
        center = image.shape
        if center[0] < 256 or center[1] < 256:
            image = self.resize_im(image)
            center = image.shape

        x = np.around((center[1]*0.5 - val*0.5), 0).astype(np.int8)
        y = np.around((center[0]*0.5 - val*0.5),0).astype(np.int8)
        return image[y:y+val, x:x+val]


    #Legacy (delete?)
    def load_image(self, path):
        #imread returns X,Y,C
        image = self.CenterCrop(cv2.imread(str(path)))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255
    
    def load_torch_image(self, path): # NEW!
        # ToTensor will convert dim from HWC to CHW, and scale from 0,255 to 0,1
        image = self.ConvertToTensor(Image.open(str(path)))
        return image
        

    def totensortorch(self, image): # THIS should now be redundant
        return torch.from_numpy(np.moveaxis(image, -1, 0)).float()


    def __getitem__(self, idx):
        if self.transform is not None:
            if self.preprocess:
                target = self.transform(self.data[idx,:])
            else:
                target = self.transform(self.Image_To_Sample_Transform(self.load_torch_image(self.OriginalImagesList[idx])))
        else:
            if self.preprocess:
                target = self.data[idx,:]
            else:
                target = self.Image_To_Sample_Transform(self.load_torch_image(self.OriginalImagesList[idx]))

        defect, mask = self.DefectGenerator(target.clone())

        return target, defect, mask




  


            
        