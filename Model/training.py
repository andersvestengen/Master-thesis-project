import torch 
from .Model import Generator_Unet1, Discriminator_1
from .Dataset import GAN_dataset
from torch.utils.data import dataloader, dataset
from torch.optim import Adam
import torchvision
from torchvision import transforms



Settings = {
    "epochs" : 20,
    "batch_size" : 16,
    "L1_loss_weight": 100,
    "lr" : 0.001,
    
    }

device = ""


training_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])


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
    training_dataloader = dataloader(dataset=GAN_dataset(
        
    ))