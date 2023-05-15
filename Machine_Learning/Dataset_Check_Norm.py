import torch
from torchvision import transforms
from Datasets.GAN_Dataset_1 import GAN_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm



Server_dir = "/home/anders/Master-thesis-project/Machine_Learning"
Preprocess_dir = "/home/anders/Thesis_image_cache"

Settings = {
            "epochs"                : 10,
            "batch_size"            : 1,
            "L1__local_loss_weight" : 50, # Don't know how much higher than 100 is stable, 300 causes issues. Might be related to gradient calc. balooning.
            "L1_loss_weight"        : 50,
            "BoxSet"               : [3,10], # min/max defect, inclusive
            "Loss_region_Box_mult"  : 1, # How many multiples of the defect box would you like the loss to account for?
            "lr"                    : 0.0002,
            "dataset_loc"           : Server_dir,
            "preprocess_storage"    : Preprocess_dir,
            "seed"                  : 172, # random training seed
            "num_workers"           : 1,
            "Data_mean"             : [0.5274, 0.4378, 0.3555],
            "Data_std"              : [0.2842, 0.2463, 0.2103],
            "Do norm"               : False, #Normalization on or off 
            "shuffle"               : True,
            "Datasplit"             : 0.8,
            "device"                : "cuda",
            "ImageHW"               : 256,
            "RestoreModel"          : False,
            #No spaces in the model name, please use '_'
            "ModelTrainingName"     : "GAN_14_Normal_Initialization_mean_0_std_0.02",
            "Drop_incomplete_batch" : True,
            "Num_training_samples"  : 18000, #Setting this to None makes the Dataloader use all available images.
            "Pin_memory"            : True
            }

norm_trans = transforms.ToTensor()

Custom_dataset = GAN_dataset(Settings, transform=None, preprocess=True)

dim_mean = torch.zeros((Settings["Num_training_samples"], 3))
dim_std = torch.zeros((Settings["Num_training_samples"], 3))

with tqdm(Custom_dataset, unit="image", leave=True) as tepoch:
    for i, image in enumerate(tepoch):
        image, _, _ = next(iter(Custom_dataset))

        dim_mean[i,:] = torch.mean(image, dim=(1,2))
        dim_std[i,:] = torch.std(image, dim=(1,2))

total_mean = torch.mean(dim_mean, dim=0)
total_std = torch.mean(dim_std, dim=0)


print("mean:", total_mean)
print("std:", total_std)
