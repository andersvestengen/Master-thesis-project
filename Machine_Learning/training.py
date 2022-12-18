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


Laptop_dir = "C:/Users/ander/Documents/Master-thesis-project/Machine_Learning/TrainingImageGenerator"
Desk_dir = "G:/Master-thesis-project/Machine_Learning"
Server_dir = "/itf-fi-ml/home/andergv/Master-thesis-project/Machine_Learning"
Preprocess_dir = "/itf-fi-ml/shared/users/andergv"

Settings = {
            "epochs"                : 100,
            "batch_size"            : 32,
            "L1_loss_weight"        : 100,
            "BoxSize"               : 5,
            "lr"                    : 0.0002,
            "dataset_loc"           : Server_dir,
            "preprocess_storage"    : Preprocess_dir,
            "seed"                  : 266, # random training seed
            "num_workers"           : 14,
            "shuffle"               : True,
            "Datahost"              : "cuda", #Should the data be located on the GPU or CPU during training?
            "Datasplit"             : 0.7,
            "device"                : "cuda",
            "ImageHW"               : 256,
            "RestoreModel"          : False,
            #No spaces in the model name, please use '_'
            "ModelTrainingName"     : "GAN_V4_New_CV2_Dataloader",
            "Drop_incomplete_batch" : True,
            "Num_training_samples"  : 25000,
            "Pin_memory"            : True
            }

# client side Settings
Settings_cli = {
            "epochs"                : 4,
            "batch_size"            : 1,
            "L1_loss_weight"        : 10,
            "BoxSize"               : 5,
            "lr"                    : 0.0002,
            "dataset_loc"           : Desk_dir,
            "preprocess_storage"    : None,
            "seed"                  : 589, # random training seed
            "num_workers"           : 0,
            "Datahost"              : "cpu", #Should the data be located on the GPU or CPU during training?
            "shuffle"               : True,
            "Datasplit"             : 0.7,
            "device"                : "cpu",
            "ImageHW"               : 256,
            "RestoreModel"          : False,
            #No spaces in the model name, please use '_'
            "ModelTrainingName"     : "LOCAL_TEST_DELETE_ME",
            "Drop_incomplete_batch" : True,
            "Num_training_samples"  : 14,
            "Pin_memory"            : False
            }

#Remove this for server training
#Settings = Settings_cli

training_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])

def weights_init(m): # from the pix2pix paper
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

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
    Generator = Generator_Unet1().to(device)
    Generator.apply(weights_init)


    # Configure dataloaders
    Custom_dataset = GAN_dataset(Settings, transform=training_transforms)

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
    GAN_loss        = torch.nn.MSELoss().to(Settings["device"])
    pixelwise_loss  = torch.nn.L1Loss().to(Settings["device"])

    Generator_optimizer = Adam(Generator.parameters(), lr=Settings["lr"], betas=[0.5, 0.999])
    Discriminator_optimizer = Adam(Discriminator.parameters(), lr=Settings["lr"]*0.5, betas=[0.5, 0.999])

    #Training
    trainer = Training_Framework(Settings, Generator, Generator_optimizer, Discriminator_optimizer, GAN_loss, pixelwise_loss, Discriminator)

    trainer.Trainer(train_loader, val_loader)