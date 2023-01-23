from torch.utils.data import DataLoader
import torch
from Models.GAN_Model_1 import Generator_Unet1, Discriminator_1
from Datasets.GAN_Dataset_1 import GAN_dataset
from Training_Framework import Model_Inference


if __name__ == '__main__':

        Settings = {
                "epochs"                : 60,
                "batch_size"            : 32,
                "L1__local_loss_weight" : 100, # Don't know how much higher than 100 is stable, 300 causes issues. Might be related to gradient calc. balooning.
                "L1_loss_weight"        : 10,
                "BoxSize"               : 10,
                "Loss_region_Box_mult"  : 1, # A multiplier based on the 'BoxSize' value. Only whole numbers!
                "lr"                    : 0.0002,
                "dataset_loc"           : None,
                "preprocess_storage"    : None,
                "seed"                  : 945, # random training seed
                "num_workers"           : 0,
                "shuffle"               : True,
                "Datasplit"             : 0.7,
                "device"                : "cuda",
                "ImageHW"               : 256,
                "RestoreModel"          : False,
                #No spaces in the model name, please use '_'
                "ModelTrainingName"     : "GAN_V6_Standard_5x5_L1loss",
                "Drop_incomplete_batch" : True,
                "Num_training_samples"  : None, #Setting this to None makes the Dataloader use all available images.
                "Pin_memory"            : False
                }

        Custom_dataset = GAN_dataset(Settings)

        imloader = DataLoader(Custom_dataset,
                                        num_workers = 1,
                                        batch_size = 1, 
                                        shuffle = True,
                                        drop_last=False)
        Model = Generator_Unet1()

        inference_run = Model_Inference(Model, imloader, Settings)
        inference_run.CreateMetrics()
        #inference_run.Inference_run()