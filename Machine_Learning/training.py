from Models.GAN_Model_1 import Generator_Unet1, UnetGenerator, init_weights
from Models.Discriminators import Discriminator_1, PixPatchGANDiscriminator, PixelDiscriminator, SpectralDiscriminator
from Models.GAN_REF_HEMIN import UNet_ResNet34
from Models.GAN_ATTN_Model import Generator_Unet_Attention, Generator_Defect_GAN
from Datasets.GAN_Dataset_1 import GAN_dataset
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from Training_Framework import Training_Framework
from torchvision import transforms
from time import time
import os
Laptop_dir = "C:/Users/ander/Documents/Master-thesis-project/Machine_Learning/TrainingImageGenerator"
#Desk_dir = "G:/Master-thesis-project/Machine_Learning"
Server_dir = "/home/anders/Master-thesis-project/Machine_Learning"
Preprocess_dir = "/home/anders/Thesis_image_cache"
Celeb_A_Dataset = "/home/anders/Celeb_A_Dataset"
Standard_training_Set = "/home/anders/Master-thesis-project/Machine_Learning/Images"
losses = ["Hinge_loss", "WGAN", "CGAN", "WGANGP"] #Choose one 
Settings = {
            "epochs"                : 4,
            "batch_size"            : 16,
            "L1__local_loss_weight" : 100, # Don't know how much higher than 100 is stable, 300 causes issues. Might be related to gradient calc. balooning.
            "L1_loss_weight"        : 100,
            "BoxSet"               : [8,8], # min/max defect, inclusive
            "Loss_region_Box_mult"  : 1, # How many multiples of the defect box would you like the loss to account for?
            "n_crit"                : 2,
            "lambda_gp"             : 10, #WGAN-GP constant
            "Blockmode"             : False, #Should the defects be random artifacts or solid chunks?
            "BlackorWhite"          : [True, False], #Whether to use black or white defects (or both)
            "CenterDefect"          : False, #This will disable the randomization of the defect within the image, and instead ensure the defect is always centered. Useful for initial training and prototyping.
            "lr"                    : 0.0004,
            "dataset_loc"           : Server_dir,
            "Loss"                  : losses[1], # Which GAN loss to train with?
            "preprocess_storage"    : Preprocess_dir,
            "seed"                  : 362, # random training seed # 172
            "num_workers"           : 4,
            "shuffle"               : True,
            "Data_mean"             : [0.3212, 0.3858, 0.2613],
            "Data_std"              : [0.2938, 0.2827, 0.2658],
            "Do norm"               : False, #Normalization on or off 
            "Datasplit"             : 0.8,
            "device"                : "cuda",
            "ImageHW"               : 128,
            "RestoreModel"          : False,
            #No spaces in the model name, please use '_'
            "ModelTrainingName"     : "InpaintingTest_Full_4_epoch",
            "Drop_incomplete_batch" : True,
            "Num_training_samples"  : None, #[None] for all available images or float [0,1] for a fraction of total images
            "Pin_memory"            : True
            }


training_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(0.2),
    transforms.RandomVerticalFlip(0.2),
])

#training_transforms = None # Removing the transforms until I'm sure they're useful.

#Try with instancenorm affine, to enable learnable parameters

def Get_name(in_list, SaveStateList):
    for model_name in in_list:
        for line in SaveStateList:
            try:
                model_arch = line.split(':')[1].strip()
            except:
                pass

            if model_name == model_arch:
                return model_arch
        else:
            continue
        break
    print("Found no compatible modelnames")

def RestoreModel(model, modeldir):
    model.load_state_dict(torch.load(modeldir, map_location=torch.device(device)))
    model.to(device)
    print("Succesfully loaded model")

if __name__ == '__main__':
    # Setup GPU (or not)
    if torch.cuda.is_available():
        Settings["device"] = "cuda"
    else:
        Settings["device"] = "cpu"
    

    device = Settings["device"]
    if input("Would you like to load a previous model[y/n] ?: ") == "y":
        models_loc = Server_dir +  "/Trained_Models"
        #Get dir list of all the current trained models
        models = os.listdir(models_loc)

        for num, model in enumerate(models):
            choice = "[" + str(num) + "]    " + model
            print(choice)

        choice  = int(input("please input modelnum: "))

        #Get model choice to be loaded 
        Generator_dir = models_loc + "/"  + models[choice] + "/model.pt"
        Discriminator_dir = models_loc + "/"  + models[choice] + "/dis_model.pt"

        modelname = models[choice]
        model_state = models_loc + "/"  + models[choice] + "/Savestate.txt"
        model_inf = []
        with open(model_state, 'r') as f:
            model_inf = [Line for Line in f]

        #Get actual model and load them 
        Gen_arch = ""
        Gen_list = ["Generator_Defect_GAN"]
        Dis_arch = ""
        Dis_list = ["PixelDiscriminator"]

        Gen_arch = Get_name(Gen_list, model_inf)
        Dis_arch = Get_name(Dis_list, model_inf)

        if Gen_arch == "Generator_Defect_GAN":
            Generator = Generator_Defect_GAN()
            RestoreModel(Generator, Generator_dir)


        if Dis_arch == "PixelDiscriminator":
            Discriminator = PixelDiscriminator()
            RestoreModel(Discriminator, Discriminator_dir)
    else:
        Discriminator = PixelDiscriminator().to(device)
        init_weights(Discriminator)
        Generator = Generator_Defect_GAN().to(device)
        init_weights(Generator)


    # Configure dataloaders
    train_dataset_file = "/home/anders/Master-thesis-project/Machine_Learning/CELEBA_Training_split"
    validation_dataset_file = "/home/anders/Master-thesis-project/Machine_Learning/CELEBA_Validation_split"

    train_set = GAN_dataset(Settings, train_dataset_file, transform=training_transforms, preprocess=False)
    val_set = GAN_dataset(Settings, validation_dataset_file, preprocess=False)


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

    Generator_optimizer = Adam(Generator.parameters(), lr=Settings["lr"]*0.5, betas=[0, 0.999])
    Discriminator_optimizer = Adam(Discriminator.parameters(), lr=Settings["lr"], betas=[0, 0.999])

    #Training
    trainer = Training_Framework(Settings, Generator, Generator_optimizer, Discriminator_optimizer, Discriminator)
    trainer.Trainer(train_loader, val_loader)