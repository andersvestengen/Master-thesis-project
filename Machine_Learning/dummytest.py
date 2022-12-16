# Checking that the Dataset actually outputs the correct pairs of items, not just at the beginning. 
from Datasets.GAN_Dataset_1 import GAN_dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
Desk_dir = "G:/Master-thesis-project/Machine_Learning"
t_dir = Desk_dir + "/output_img/Training"
v_dir = Desk_dir + "/output_img/Validation"

Settings = {
            "epochs"                : 40,
            "batch_size"            : 1,
            "L1_loss_weight"        : 100,
            "BoxSize"               : 5,
            "lr"                    : 0.0002,
            "dataset_loc"           : Desk_dir,
            "preprocess_storage"    : None,
            "seed"                  : 266, # random training seed
            "num_workers"           : 1,
            "shuffle"               : True,
            "Datahost"              : "cpu", #Should the data be located on the GPU or CPU during training?
            "Datasplit"             : 0.7,
            "device"                : "cpu",
            "ImageHW"               : 256,
            "RestoreModel"          : False,
            #No spaces in the model name, please use '_'
            "ModelName"             : "GAN_V3_Box_30",
            "Drop_incomplete_batch" : True,
            "Num_training_samples"  : None,
            "Pin_memory"            : False
            }

training_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])
Custom_dataset = GAN_dataset(Settings, transform=training_transforms)