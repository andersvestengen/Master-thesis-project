from Models.GAN_Model_1 import Discriminator_1, Generator_Unet1
from Models.GAN_REF_HEMIN import UNet_ResNet34
from Datasets.GAN_Dataset_1 import GAN_dataset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Adam
from Training_Framework import Training_Framework
from torchvision import transforms
from time import time

Laptop_dir = "C:/Users/ander/Documents/Master-thesis-project/Machine_Learning/TrainingImageGenerator"
#Desk_dir = "G:/Master-thesis-project/Machine_Learning"
Server_dir = "/home/anders/Master-thesis-project/Machine_Learning"
Preprocess_dir = "/home/anders/Thesis_image_cache"

Settings = {
            "epochs"                : 20,
            "batch_size"            : 16,
            "L1__local_loss_weight" : 100, # Don't know how much higher than 100 is stable, 300 causes issues. Might be related to gradient calc. balooning.
            "L1_loss_weight"        : 100,
            "BoxSet"               : [3,10], # min/max defect, inclusive
            "Loss_region_Box_mult"  : 1, # How many multiples of the defect box would you like the loss to account for?
            "lr"                    : 0.0002,
            "dataset_loc"           : Server_dir,
            "preprocess_storage"    : Preprocess_dir,
            "seed"                  : 172, # random training seed
            "num_workers"           : 4,
            "shuffle"               : True,
            "Data_mean"             : [0.3212, 0.3858, 0.2613],
            "Data_std"              : [0.2938, 0.2827, 0.2658],
            "Do norm"               : True, #Normalization on or off 
            "Datasplit"             : 0.8,
            "device"                : "cuda",
            "ImageHW"               : 256,
            "RestoreModel"          : False,
            #No spaces in the model name, please use '_'
            "ModelTrainingName"     : "GAN_15_ImageNet_norm_20_epoch",
            "Drop_incomplete_batch" : True,
            "Num_training_samples"  : None, #Setting this to None makes the Dataloader use all available images.
            "Pin_memory"            : True
            }


training_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(0.2),
    transforms.RandomVerticalFlip(0.2),
])

#training_transforms = None # Removing the transforms until I'm sure they're useful.

#Try with instancenorm affine, to enable learnable parameters

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 1)


if __name__ == '__main__':
    # Setup GPU (or not)
    if torch.cuda.is_available():
        Settings["device"] = "cuda"
    else:
        Settings["device"] = "cpu"
    

    device = Settings["device"]

    #Load models
    Discriminator = Discriminator_1().to(device)
    Discriminator.apply(weights_init)
    #Generator = UNet_ResNet34().to(device)
    Generator = Generator_Unet1().to(device)
    Generator.apply(weights_init)


    # Configure dataloaders
    Custom_dataset = GAN_dataset(Settings, transform=training_transforms, preprocess=False)

    Settings["Dataset_name"] = Custom_dataset.name 
    dataset_len = len(Custom_dataset)
    train_split = int(dataset_len*Settings["Datasplit"])
    val_split = int(dataset_len - train_split)

    train_set, val_set = torch.utils.data.random_split(Custom_dataset, [train_split, val_split])

    train_loader = DataLoader(train_set,
                            num_workers     = Settings["num_workers"],
                            batch_size      = Settings["batch_size"], 
                            shuffle         = Settings["shuffle"],
                            drop_last       = Settings["Drop_incomplete_batch"],
                            pin_memory      = Settings["Pin_memory"])


    val_loader = DataLoader(val_set,
                            num_workers     = Settings["num_workers"],
                            batch_size      = Settings["batch_size"], 
                            shuffle         = Settings["shuffle"],
                            drop_last       = Settings["Drop_incomplete_batch"],
                            pin_memory      = Settings["Pin_memory"])
    
    # Loss functions
    GAN_loss        = torch.nn.MSELoss().to(Settings["device"]) # GAN loss for GEN and DIS
    pixelwise_loss  = torch.nn.L1Loss().to(Settings["device"]) # loss for the local patch around the defect

    Generator_optimizer = Adam(Generator.parameters(), lr=Settings["lr"], betas=[0.5, 0.999])
    Discriminator_optimizer = Adam(Discriminator.parameters(), lr=Settings["lr"]*0.5, betas=[0.5, 0.999])

    #Training
    trainer = Training_Framework(Settings, Generator, Generator_optimizer, Discriminator_optimizer, GAN_loss, pixelwise_loss, Discriminator)
    trainer.Trainer(train_loader, val_loader)