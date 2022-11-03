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

# Calculate output of image discriminator (PatchGAN)
patch = (1, 512 // 2 ** 4, 512 // 2 ** 4)


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
            }



training_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])


#TODO: add numpy load/store functionality and complete the analytics section at the bottom of the training loop.
Generator_loss_train = np.zeros(Settings["epochs"])
Discriminator_loss_train = np.zeros(Settings["epochs"])
Generator_loss_validation = np.zeros(Settings["epochs"])
Discriminator_loss_validation = np.zeros(Settings["epochs"])

def Display_graphs(in1, in2, in3, in4, epochs):
    xaxis = np.arange(0, epochs+1)
    plt.plot(xaxis, in1)
    plt.plot(xaxis, in2)
    plt.plot(xaxis, in3)    
    plt.plot(xaxis, in4)
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

    def validation_sampler(epoch):
        print("Running validation sample")
        with torch.no_grad():
            Generator.eval()
            Discriminator.eval()
            inputs, targets = torch.utils.data.RandomSampler(val_loader)
            Gen_faulty_image = inputs
            True_output_image = targets
            
            # Adversarial ground truths
            valid = Tensor(np.ones((Gen_faulty_image.size(0), *patch))).requires_grad=False
            fake = Tensor(np.zeros((Gen_faulty_image.size(0), *patch))).requires_grad=False                       
            
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
    
    for epoch in range(1, Settings["epochs"] + 1):
        # Look to add something for lowering the learning rate after a number of epochs
            
        
        # Add a try except block for more robust functionality.
        
        print("Training on epoch: ", epoch)
        mean_loss = 0
        # Training loop
        for i, (inputs, targets) in tqdm(enumerate(train_loader)):
            
            print("on input", i, "input has size:", inputs.size())
            #Model inputs
            Gen_faulty_image = inputs
            True_output_image = targets
            
            # Adversarial ground truths
            valid = Tensor(np.ones((Gen_faulty_image.size(0), *patch))).requires_grad=False
            fake = Tensor(np.zeros((Gen_faulty_image.size(0), *patch))).requires_grad=False
            
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
            
            Total_loss_Discriminator.backwards()
            
            Discriminator_optimizer.step()
            
            
            #Analytics
            Generator_loss_train[epoch] = Total_loss_Generator
            Discriminator_loss_train[epoch] = Total_loss_Discriminator
            print("Loss for epoch:", epoch, "was: (Generator loss)", Total_loss_Generator, "(Discriminator) ", Total_loss_Discriminator)
        
        # Validation loop
        validation_sampler(epoch)
    Display_graphs(Generator_loss_train, Generator_loss_validation, Discriminator_loss_train, Discriminator_loss_validation)


if __name__ == '__main__':
    main()