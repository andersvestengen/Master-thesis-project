import random
import torch
#from Datasets.Dataset import DCGAN_dataset
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from Datasets.GAN_Dataset_1 import GAN_dataset
from Models.GAN_Model_1 import init_weights, Generator_Unet1, PixPatchGANDiscriminator
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from datetime import datetime
import os
from time import time
from PIL import Image
import sys

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
Server_dir = "/home/anders/Master-thesis-project/Machine_Learning"
# Init settings

Settings = {

            "dataset_loc"           :   Server_dir,
            "ModelTrainingName"     :   "SanityGAN",
            "dataroot"              :   'Images',
            "L1_lambda"             :   100,
            "device"                :   "cuda",
            "Loss_region_Box_mult"  :   1,
            "BoxSet"                :   [3,10], # min/max defect, inclusive
            "num_workers"           :   2,      #Num memory workers
            "batch_size"            :   1,    #Num images in training batch
            "image_size"            :   256,     #Image H,W will be resized to this value
            "num_channels"          :   3,      #num image channels
            "seed"                  :   327,
            "Num_training_samples"  :   None,
            "z_size"                :   100,    #size of latent z-vector
            "ngf"                   :   64,     #size of feature maps in generator
            "ndf"                   :   64,     #size of feature maps in discriminator
            "epochs"                :   3,      #Number of epochs
            "Data_mean"             :   [0.3212, 0.3858, 0.2613],
            "Data_std"              :   [0.2938, 0.2827, 0.2658],
            "Do norm"               :   False, #Normalization on or off 
            "lr"                    :   0.0002, #Learning rate
            "beta1"                 :   0.5,    #Beta 1 parameter for optimizer
            "n_gpu"                 :   1,      #Number of GPU's [0 -> CPU training]
}

#Create the directory of the model (Look back at this for batch training.)
times = str(datetime.now())
stamp = times[:-16] + "_" + times[11:-7].replace(":", "-")
Modeldir = ""
if Settings["ModelTrainingName"] is not None:
    Modeldir = Settings["dataset_loc"] +  "/Trained_Models/" + Settings["ModelTrainingName"] + " (" + stamp +")"
else:
    Modeldir = Settings["dataset_loc"] + "/Trained_Models/" + "GAN_Model" + " " + stamp

os.makedirs(Modeldir)
os.makedirs(Modeldir + "/" + "Images")
print("saving model to:", Modeldir)


def PIL_concatenate_h(im1, im2):
    out = Image.new('RGB', (im1.width + im2.width, im1.height))
    out.paste(im1, (0,0))
    out.paste(im2, (im1.width, 0))
    return out

def FromTorchTraining(image):
    #Returns a trainable tensor back into a visual image.

    return image.permute(1,2,0).mul_(255).clamp(0,255).to("cpu", torch.uint8).numpy()


def Make_Label_Tensor(tensor_size, bool_val):
    """
    Return a label tensor with size tensor_size and values bool_val
    """       

    if bool_val:
        label_tensor = torch.tensor(1, device=device, dtype=torch.float32, requires_grad=False)
    else:
        label_tensor = torch.tensor(0, device=device, dtype=torch.float32, requires_grad=False)

    return label_tensor.expand_as(tensor_size)


def Save_Model():
            torch.save(netG.state_dict(), str(Modeldir + "/model.pt"))



def backward_D(real_B, real_A, fake_B):
    """Calculate GAN loss for the discriminator"""
    # Fake; stop backprop to the generator by detaching fake_B
    fake_AB = torch.cat((real_A, fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
    pred_fake = netD(fake_AB.detach())
    fake = Make_Label_Tensor(pred_fake, False)
    loss_D_fake = criterion(pred_fake, fake)
    # Real
    real_AB = torch.cat((real_A, real_B), 1)
    pred_real = netD(real_AB)
    valid = Make_Label_Tensor(pred_real, True)
    loss_D_real = criterion(pred_real, valid)
    # combine loss and calculate gradients
    loss_D = (loss_D_fake + loss_D_real) * 0.5
    loss_D.backward()
    return loss_D.item(), pred_real.mean().item(), pred_fake.mean().item()

def backward_G(real_B, real_A, fake_B):
    """Calculate GAN and L1 loss for the generator"""
    # First, G(A) should fake the discriminator
    fake_AB = torch.cat((real_A, fake_B), 1)
    pred_fake = netD(fake_AB)
    valid = Make_Label_Tensor(pred_fake, True)
    loss_G_GAN = criterion(pred_fake, valid)
    # Second, G(A) = B
    loss_G_L1 = Pixelloss(fake_B, real_B) * Settings["L1_lambda"]
    # combine loss and calculate gradients
    loss_G = loss_G_GAN + loss_G_L1
    loss_G.backward()
    return loss_G.item(), pred_fake.mean().item(), loss_G_L1.item()


def optimize_parameters(real_A, real_B):
    fake_B = netG(real_A)
    
    # update D
    netD.requires_grad_(True)                    # enable backprop for D
    optimizerD.zero_grad()                                      # set D's gradients to zero
    loss_D, Dx, D_G_z1 = backward_D(real_B, real_A, fake_B)          # calculate gradients for D
    optimizerD.step()                           # update D's weights
    
    # update G
    netD.requires_grad_(False)                   # D requires no gradients when optimizing G
    optimizerG.zero_grad()                      # set G's gradients to zero
    loss_G, D_G_z2, G_pixelloss = backward_G(real_B, real_A, fake_B) # calculate graidents for G
    optimizerG.step()                           # update G's weights
    return loss_G, loss_D, Dx, D_G_z1, D_G_z2, G_pixelloss


#Creating the dataset (not in a separate file this time)

dataset = GAN_dataset(Settings, transform=None, preprocess=False)


# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=Settings["batch_size"],
                                         shuffle=True, num_workers=Settings["num_workers"])

# Decide which device we want to run on
device = torch.device("cuda" if (torch.cuda.is_available() and Settings["n_gpu"] > 0) else "cpu")

# Create and initalize the models
netG = Generator_Unet1().to(device)
netD = PixPatchGANDiscriminator(norm_layer=nn.InstanceNorm2d).to(device)
init_weights(netG)
init_weights(netD)


#Loss functions and optimizers

# Initialize the ``BCELoss`` function
criterion = nn.BCEWithLogitsLoss().to(device)
Pixelloss = nn.L1Loss().to(device)

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=Settings["lr"], betas=(Settings["beta1"], 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=Settings["lr"], betas=(Settings["beta1"], 0.999))

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = np.zeros(Settings["epochs"]*len(dataloader))
G_pixellosses = np.zeros(Settings["epochs"]*len(dataloader))
D_losses = np.zeros(Settings["epochs"]*len(dataloader))
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(Settings["epochs"]):
        # For each batch in the dataloader
        epoch_s_t = time()
        for i, data in enumerate(dataloader, 0):


            real_im, defect_im, _ = data

            real_B = real_im.to(device)
            real_A = defect_im.to(device)

            loss_G, loss_D, D_x, D_G_z1, D_G_z2, G_pixelloss = optimize_parameters(real_B, real_A) # GAN optimizer function

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, Settings["epochs"], i, len(dataloader),
                        loss_D, loss_G, D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses[iters] = loss_G
            D_losses[iters] = loss_D
            G_pixellosses[iters] = G_pixelloss

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == Settings["epochs"]-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    netG.eval()
                    if real_A.size(0) > 1:
                        real_im = real_A[0,:,:,:].clone()
                    else:
                         real_im = real_A.clone()
                    fake_B = netG(real_im)
                    im = Image.fromarray(FromTorchTraining(fake_B.squeeze(0)))
                    co = Image.fromarray(FromTorchTraining(real_im.squeeze(0)))
                    PIL_concatenate_h(co, im).save(Modeldir + "/" + "Images/" + "Image_" + str(iters) + ".jpg", "JPEG")
                    netG.train()

            if (epoch == 0 and iters == 0) or (G_losses[iters] < np.amin(G_losses[:iters])):
                Save_Model()
                 
            iters += 1
        epoch_time = (int(time() - epoch_s_t) / 60)
        print("finished epoch", epoch, "training took:", epoch_time, "minutes")
        print("estimated time left:", (Settings["epochs"] - epoch) * epoch_time, "minutes")


#Post training metrics

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(Modeldir + "/GAN_loss.png")
plt.show()


plt.figure(figsize=(10,5))
plt.title("Generator pixelloss")
plt.plot(G_pixellosses,label="G")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(Modeldir + "/Generator_pixelloss.png")
plt.show()