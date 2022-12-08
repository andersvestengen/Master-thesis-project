from torch.utils.data import DataLoader
import torch
from Models.GAN_Model_1 import Generator_Unet1, Discriminator_1
from Datasets.GAN_Dataset_1 import GAN_dataset
from Training_Framework import Model_Inference


if __name__ == '__main__':
    Custom_dataset = GAN_dataset(seed=676)

    imloader = DataLoader(Custom_dataset,
                                    num_workers = 1,
                                    batch_size = 1, 
                                    shuffle = True,
                                    drop_last=False)
    Model = Generator_Unet1()

    inference_run = Model_Inference(Model, imloader)