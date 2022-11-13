from random import shuffle
import torch 
from Models.GAN_Model_1 import Generator_Unet1, Discriminator_1
from Datasets.GAN_Dataset_1 import GAN_dataset
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torchvision
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from sys import exit

#Defining random seeds
seed_num = 23
gan_gen = torch.manual_seed(seed_num)
random.seed(seed_num)
np.random.seed(seed_num)

"""
TODO:
    - Complete Numpy load/store functionality
    - Add a try/except block to the training loop
    - Add a way to save the model during/after training.

"""

Laptop_dir = "C:/Users/ander/Documents/Master-thesis-project/Machine Learning/TrainingImageGenerator"
Desk_dir = "G:/Master-thesis-project/Machine Learning/TrainingImageGenerator"
Server_dir = "/itf-fi-ml/home/andergv/Master-thesis-project/Machine Learning/TrainingImageGenerator"
Preprocess_dir = "/itf-fi-ml/shared/users/andergv"

# Need to add os.getcwd() to dataset_loc or figure out something similar.
Settings = {
            "epochs"                : 50,
            "batch_size"            : 16,
            "L1_loss_weight"        : 100,
            "lr"                    : 0.001,
            "dataset_loc"           : Server_dir,
            "preprocess_storage"    : Preprocess_dir,
            "num_workers"           : 1,
            "shuffle"               : True,
            "Datasplit"             : 0.7,
            "Device"                : "cpu",
            "ImageHW"               : 256,
            "RestoreModel"          : False,
            "ModelName"             : "GAN_1_best.pt",
            "preprocess"            : True,
            "Drop_incomplete_batch" : True,
            "Num_training_samples"  : 1500,
            }



# Calculate output of image discriminator (PatchGAN)
patch = (1, Settings["ImageHW"] // 2 ** 4, Settings["ImageHW"] // 2 ** 4)

training_transforms = transforms.Compose([
    transforms.CenterCrop(Settings["ImageHW"]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])


#TODO: add numpy load/store functionality and complete the analytics section at the bottom of the training loop.
Generator_loss_train = np.zeros(Settings["epochs"])
Discriminator_loss_train = np.zeros(Settings["epochs"])
Generator_loss_validation = np.zeros(Settings["epochs"])
Discriminator_loss_validation = np.zeros(Settings["epochs"])



def Display_graphs(in1, in2, in3, in4):
    #Display_graphs(Generator_loss_train, Generator_loss_validation, Discriminator_loss_train, Discriminator_loss_validation)
    xaxis = np.arange(0, in1.shape[0])
    plt.plot(xaxis, in1, label="Generator loss training")
    plt.plot(xaxis, in2, label="Generator loss validation")
    plt.plot(xaxis, in3, label="Discriminator loss training")    
    plt.plot(xaxis, in4, label="Discriminator loss validation")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def SaveNPY(*args):
    np.savez('Analytics.npz', *args)
     
    
def main():
    # Setup GPU (or not)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Loss functions
    GAN_loss        = torch.nn.MSELoss().to(device)
    pixelwise_loss  = torch.nn.L1Loss().to(device)

    # Models
    Generator       = Generator_Unet1().to(device)
    Discriminator   = Discriminator_1().to(device)
    
        
    #Restore previous model
    if Settings["RestoreModel"]:
        checkpoint = torch.load(str( Settings["dataset_loc"] + "/" +  Settings["ModelName"] ), map_location=device)
        Generator.load_state_dict(checkpoint["Generator_state_dict"])
        Discriminator.load_state_dict(checkpoint["Discriminator_state_dict"])
        
        Generator.to(device)
        Discriminator.to(device)
        print("Succesfully loaded previous model")
    
    # Optimizers
    Generator_optimizer = Adam(Generator.parameters(), lr=Settings["lr"])
    Discriminator_optimizer = Adam(Discriminator.parameters(), lr=Settings["lr"])
    

    # Configure dataloaders
    Custom_dataset = GAN_dataset(preprocess_storage=Settings["preprocess_storage"], training_samples=Settings["Num_training_samples"], seed=seed_num, workingdir=Settings["dataset_loc"], transform=training_transforms, preprocess=Settings["preprocess"])

    dataset_len = len(Custom_dataset)

    train_split = int(dataset_len*Settings["Datasplit"])
    val_split = int(dataset_len - train_split)
    
    train_set, val_set = torch.utils.data.random_split(Custom_dataset, [train_split, val_split])
    
    train_loader = DataLoader(train_set,
                                   num_workers = Settings["num_workers"],
                                   batch_size = Settings["batch_size"], 
                                   shuffle = Settings["shuffle"],
                                   drop_last=Settings["Drop_incomplete_batch"])

    val_loader = DataLoader(val_set,
                                   num_workers = Settings["num_workers"],
                                   batch_size = Settings["batch_size"], 
                                   shuffle = Settings["shuffle"],
                                   drop_last=Settings["Drop_incomplete_batch"])   
    # Tensor type (Do I need this?)
    #Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor




    valid = torch.ones((Settings["batch_size"], *patch)).to(device)# These both used to have requires_grad=False, but that seems to force-cast this to a bool variable which produces errors.
    fake = torch.zeros((Settings["batch_size"], *patch)).to(device)
    
    def validation_sampler(epoch):
        Gen_loss_avg = 0
        Disc_loss_avg = 0
        with torch.no_grad():
            with tqdm(val_loader, unit='batch',position=1, leave=True) as vepoch:
                Generator.eval()
                Discriminator.eval()
                
                for inputs, targets in vepoch:
                    tepoch.set_description(f"Validation on Epoch {epoch}/{Settings['epochs']}")
                    
                    Gen_faulty_image = inputs.to(device)
                    True_output_image = targets.to(device)
                    
                    # Adversarial ground truths
                    #valid = Tensor(np.ones((Gen_faulty_image.size(0), *patch))).float()
                    #fake = Tensor(np.zeros((Gen_faulty_image.size(0), *patch))).float()                    
                    
                    # Generator loss            
                    Generated_output_image = Generator(Gen_faulty_image)
                    predict_fake = Discriminator(Generated_output_image, Gen_faulty_image)
                    loss_GAN = GAN_loss(predict_fake, valid)
                    #Pixelwise loss
                    loss_pixel = pixelwise_loss(Generated_output_image, True_output_image) # might be misinterpreting the loss inputs here.
                    
                    #Total loss
                    Total_loss_Generator = loss_GAN + Settings["L1_loss_weight"] * loss_pixel

                    predicted_real = Discriminator(True_output_image, Gen_faulty_image)
                    loss_real = GAN_loss(predicted_real, valid)
                    
                    # Fake loss
                    predict_fake = Discriminator(Generated_output_image.detach(), True_output_image)
                    loss_fake = GAN_loss(predict_fake, fake)
                    # Total loss
                    Total_loss_Discriminator = 0.5 * (loss_real + loss_fake)
                    
                    # Analytics
                    Gen_loss_avg += Total_loss_Generator.item()
                    Disc_loss_avg += Total_loss_Discriminator.item()
                
                    vepoch.set_postfix(Gen_loss_val = (Gen_loss_avg / val_split), Disc_loss_val = (Disc_loss_avg / val_split), Validation_run_on_epoch=epoch)
                
            Generator_loss_validation[epoch] = Gen_loss_avg / val_split
            Discriminator_loss_validation[epoch] = Disc_loss_avg / val_split  
    

        
    for epoch in range(Settings["epochs"]):
        # Look to add something for lowering the learning rate after a number of epochs
        # Add a try except block for more robust functionality.
        
        mean_loss = 0
        Gen_loss_avg = 0
        Disc_loss_avg = 0
        # Training loop
        with tqdm(train_loader, unit='batch', leave=False) as tepoch:
            for inputs, targets in tepoch:
                tepoch.set_description(f"Training on Epoch {epoch}/{Settings['epochs']}")
                if epoch > 0:
                    tepoch.set_description(f"Epoch {epoch}/{Settings['epochs']} Gen_loss {Generator_loss_train[epoch-1]:.5f} Disc_loss {Discriminator_loss_train[epoch-1]:.5f}")
 
                #Model inputs
                
                Gen_faulty_image = inputs.to(device)
                True_output_image = targets.to(device)
                

                #------ Train the Generator
                Generator_optimizer.zero_grad()
                
                # Generator loss            
                Generated_output_image = Generator(Gen_faulty_image)
                predict_fake = Discriminator(Generated_output_image, Gen_faulty_image)
                loss_GAN = GAN_loss(predict_fake, valid)
                #Pixelwise loss
                loss_pixel = pixelwise_loss(Generated_output_image, True_output_image) # might be misinterpreting the loss inputs here.
                
                #Total loss
                Total_loss_Generator = loss_GAN + Settings["L1_loss_weight"] * loss_pixel
                
            
                
                Total_loss_Generator.backward()

                Generator_optimizer.step()
                
                
                #------ Train Discriminator
                
                Discriminator_optimizer.zero_grad()
                
                # Real loss 
                predicted_real = Discriminator(True_output_image, Gen_faulty_image)
                loss_real = GAN_loss(predicted_real, valid)
                
                # Fake loss
                predict_fake = Discriminator(Generated_output_image.detach(), True_output_image)
                loss_fake = GAN_loss(predict_fake, fake)
                # Total loss
                Total_loss_Discriminator = 0.5 * (loss_real + loss_fake)
                
                Total_loss_Discriminator.backward()
                
                Discriminator_optimizer.step()
                
                
                #Analytics
                Gen_loss_avg += Total_loss_Generator.item()
                Disc_loss_avg += Total_loss_Discriminator.item()
                
                
            Generator_loss_train[epoch] = Gen_loss_avg / train_split
            Discriminator_loss_train[epoch] = Disc_loss_avg / train_split          
            # Validation loop
            validation_sampler(epoch)
            #Update model if validation score increased
            if (epoch > 0) and ( Generator_loss_validation[epoch] < Generator_loss_validation[epoch-1] ):
                torch.save({
                    "Generator_state_dict"      :   Generator.state_dict(),
                    "Discriminator_state_dict"  :   Discriminator.state_dict(),
                }, str( Settings["dataset_loc"] + "/" +  Settings["ModelName"] ))
    if device == "cpu":
        Display_graphs(Generator_loss_train, Generator_loss_validation, Discriminator_loss_train, Discriminator_loss_validation)
    SaveNPY(Generator_loss_train, Generator_loss_validation, Discriminator_loss_train, Discriminator_loss_validation)


if __name__ == '__main__':
    main()