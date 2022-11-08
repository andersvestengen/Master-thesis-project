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
    - Add tqdm loading bar to the training loop
    - Add something to figure out the location of the 'dataset_loc' in the Settings Dict.
    - Complete Numpy load/store functionality
    - Add a try/except block to the training loop
    - Add a way to save the model during/after training.


    - Test the whole thing on a laptop or computer (home computer), and then on the server
"""



# Need to add os.getcwd() to dataset_loc or figure out something similar.
Settings = {
            "epochs"            : 10,
            "batch_size"        : 1,
            "L1_loss_weight"    : 100,
            "lr"                : 0.001,
            "dataset_loc"       : "C:/Users/ander/Documents/Master-thesis-project/Machine Learning/TrainingImageGenerator",#"G:/Master-thesis-project/Machine Learning/TrainingImageGenerator",
            "num_workers"       : 1,
            "shuffle"           : True,
            "Datasplit"         : 0.7,
            "Device"            : "cpu",
            "ImageHW"           : 256,
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

def main():

    # Loss functions
    GAN_loss        = torch.nn.MSELoss()
    pixelwise_loss  = torch.nn.L1Loss()

    # Models
    Generator       = Generator_Unet1()
    Discriminator   = Discriminator_1()
    
    
    # Setup GPU (or not)
    if torch.cuda.is_available():
        device = "cuda"
        Generator = Generator().cuda()
        Discriminator = Discriminator().cuda()
    else:
        device = "cpu"
        
    
    # Optimizers
    Generator_optimizer = Adam(Generator.parameters(), lr=Settings["lr"])
    Discriminator_optimizer = Adam(Discriminator.parameters(), lr=Settings["lr"])
    

    # Configure dataloaders
    Custom_dataset = GAN_dataset(workingdir=Settings["dataset_loc"], transform=training_transforms)

    dataset_len = len(Custom_dataset)

    train_split = int(dataset_len*Settings["Datasplit"])
    val_split = int(dataset_len - train_split)
    
    train_set, val_set = torch.utils.data.random_split(Custom_dataset, [train_split, val_split])
    
    train_loader = DataLoader(train_set,
                                   num_workers = Settings["num_workers"],
                                   batch_size = Settings["batch_size"], 
                                   shuffle = Settings["shuffle"])

    val_loader = DataLoader(val_set,
                                   num_workers = Settings["num_workers"],
                                   batch_size = Settings["batch_size"], 
                                   shuffle = Settings["shuffle"])   
    # Tensor type (Do I need this?)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


    #Adversarial ground truths
    valid = Tensor(torch.ones((Settings["batch_size"], *patch))).to(torch.device(device))# These both used to have requires_grad=False, but that seems to force-cast this to a bool variable which produces errors.
    fake = Tensor(torch.zeros((Settings["batch_size"], *patch))).to(torch.device(device))
    
    def validation_sampler(epoch):
        with torch.no_grad():
            Generator.eval()
            Discriminator.eval()
            inputs, targets  = next(iter(val_loader))
            Gen_faulty_image = inputs
            True_output_image = targets
            
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

            Generator_loss_validation[epoch] = Total_loss_Generator
            Discriminator_loss_validation[epoch] = Total_loss_Discriminator  
    
    
    for epoch in range(Settings["epochs"]):
        # Look to add something for lowering the learning rate after a number of epochs
        # Add a try except block for more robust functionality.
        
        mean_loss = 0
        Gen_loss_avg = 0
        Disc_loss_avg = 0
        # Training loop
        with tqdm(train_loader, unit='"batch', leave=False) as tepoch:
            for inputs, targets in tepoch:
                tepoch.set_description(f"Epoch {epoch}/{Settings['epochs']}")
                if epoch > 0:
                    tepoch.set_description(f"Epoch {epoch}/{Settings['epochs']} Gen_loss {Generator_loss_train[epoch]:.5f} Disc_loss {Discriminator_loss_train[epoch]:.5f}")
               
                #Model inputs
                
                Gen_faulty_image = inputs
                True_output_image = targets
                
                #print("This is the size:", valid.size(), fake.size)
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
                #print("Loss for epoch:", epoch, "was: (Generator loss)", Total_loss_Generator, "(Discriminator) ", Total_loss_Discriminator)
                
            Generator_loss_train[epoch] = Gen_loss_avg / Settings["batch_size"]
            Discriminator_loss_train[epoch] = Disc_loss_avg / Settings["batch_size"]          
            # Validation loop
            validation_sampler(epoch)
    print("These are the shapes for the training analytics:")
    print(Generator_loss_train, Generator_loss_validation, Discriminator_loss_train, Discriminator_loss_validation)
    Display_graphs(Generator_loss_train, Generator_loss_validation, Discriminator_loss_train, Discriminator_loss_validation)


if __name__ == '__main__':
    main()