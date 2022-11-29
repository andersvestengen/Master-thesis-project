from torchvision import transforms
import torch 
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt 
import sys
sys.path.insert(0, "G:/Master-thesis-project/Machine_Learning/Models")
sys.path.insert(0, "G:/Master-thesis-project/Machine_Learning/Datasets")
from GAN_Model_1 import Generator_Unet1
from GAN_Dataset_1 import GAN_dataset
from PIL import Image




BoxSize = 5
boolean_sample = [True, False]

ToPILImageTrans = transforms.ToPILImage()

Settings = {
            "ImageHW"           : 256,


            }

# May need to also scale the output Tensor -> PILImage manually
training_transforms = transforms.Compose([
    transforms.CenterCrop(Settings["ImageHW"]),
    transforms.ToTensor()
])

def getSample(sampleinput):
    """
    returns a random sample between the minimum Boxsize and the sampleInput (height/width)
    """

    return int( ( sampleinput - BoxSize ) * np.random.random_sample())

def DefectGenerator(imageMatrix):
    """
    Takes a matrix-converted image and returns a training sample of that image with a randomly degraded [Boxsize * Boxsize] square, and coordinates for loss function.
    
    """
    ImageHeight = imageMatrix.shape[1]
    ImageWidth = imageMatrix.shape[2]
    SampleH = getSample(ImageHeight)
    SampleW = getSample(ImageWidth)
    intSample = imageMatrix[:,SampleH:SampleH + BoxSize,SampleW:SampleW + BoxSize] 
    mask = np.random.choice(boolean_sample, p=[0.8, 0.2], size=(intSample.shape[1:]))
    r = np.full((intSample.shape), 0)
    intSample[:,mask] = r[:,mask] 
    imageMatrix[:, SampleH:SampleH+BoxSize,SampleW:SampleW+BoxSize] = intSample
    
    return imageMatrix


def Display_graphs(npyname):
    loaded_arrays = np.load(npyname)
    xaxis = np.arange(0, loaded_arrays['arr_0'].shape[0])
    plt.plot(xaxis, loaded_arrays['arr_0'], label="Generator loss training")
    plt.plot(xaxis, loaded_arrays['arr_1'], label="Generator loss validation")
    plt.plot(xaxis, loaded_arrays['arr_2'], label="Discriminator loss training")    
    plt.plot(xaxis, loaded_arrays['arr_3'], label="Discriminator loss validation")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("Training_analytics.png")

def RestoreModel(input):
    checkpoint = torch.load(input, map_location=torch.device('cpu'))
    Generator.load_state_dict(checkpoint["Generator_state_dict"])
    print("Succesfully loaded previous model")

def Inference_run(image):
    Generator.eval()
    Generated_output_image = Generator(image)
    im = ToPILImageTrans(image.squeeze(0))
    im.save("Defectgeneratedsample.jpg")
    output = ToPILImageTrans(Generated_output_image.squeeze(0))
    output.save("Reconstructed_Image.jpg")


if __name__ == '__main__':
    # Define strings and models

    Desk_GAN1_dir = "G:/Thesis_models_and_data/GAN_1_better_parameters/"
    npy_store_dir = "Analytics.npz"
    GAN1_model_dir = "GAN_1_best.pt"

    #Blank model
    Generator = Generator_Unet1()

    # Datasets and loaders
    Desk_dir = "G:/Master-thesis-project/Machine_Learning"
    Expset = GAN_dataset(preprocess_storage=None, training_samples=1500, seed=18, workingdir=Desk_dir, transform=training_transforms)

    train_loader = DataLoader(Expset,
                                    num_workers = 1,
                                    batch_size = 1, 
                                    shuffle = True,
                                    drop_last=True)

    datasetimage, sample = next(iter(train_loader))
    # Load model and analytics
    Display_graphs((Desk_GAN1_dir + npy_store_dir))
    RestoreModel((Desk_GAN1_dir + GAN1_model_dir))

    # Do Inference on image
    Inference_run(sample)