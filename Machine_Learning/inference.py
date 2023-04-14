from torch.utils.data import DataLoader
import torch
from Models.GAN_Model_1 import Generator_Unet1, Discriminator_1
from Models.GAN_REF_HEMIN import UNet_ResNet34
from Datasets.GAN_Dataset_1 import GAN_dataset
from Training_Framework import Model_Inference
import os
import shutil

if __name__ == '__main__':

		Machine_learning_dir = "/home/anders/Master-thesis-project/Machine_Learning" # should point to the Machine learning folder of the local directory
		Preprocess_dir = "/home/anders/Thesis_image_cache" # This can be set wherever you want. This is where the dataloader will create the image cache after converting the training images to tensor files.
		#No editing of the images happens in the step above, its simply the dataloader converting the original images to tensors so the whole training set can be loaded in RAM before training starts. 

		Settings = {
				"epochs"                : 5,
				"batch_size"            : 1,
				"L1__local_loss_weight" : 50, # Don't know how much higher than 100 is stable, 300 causes issues. Might be related to gradient calc. balooning.
				"L1_loss_weight"        : 50,
				"BoxSet"               : [3,10], # Low/high Boxsize of the error region. The value represents for length and with.
				"Loss_region_Box_mult"  : 3, # This is now static at 3, do not change!
				"lr"                    : 0.0002,
				"dataset_loc"           : Machine_learning_dir,
				"preprocess_storage"    : Preprocess_dir,
				"seed"                  : 172, # random training seed
				"num_workers"           : 1,
				"shuffle"               : True,
				"Datasplit"             : 0.7,
				"device"                : "cuda",
				"ImageHW"               : 256,
				"RestoreModel"          : False,
				#No spaces in the model name, please use '_'
				"ModelTrainingName"     : "RESOURCE_TEST_DELETE_ME",
				"Drop_incomplete_batch" : True,
				"Num_training_samples"  : 15000, #Setting this to None makes the Dataloader use all available images.
				"Pin_memory"            : True
				}

		Custom_dataset = GAN_dataset(Settings, preprocess=True)

		imloader = DataLoader(Custom_dataset,
										num_workers = 0,
										batch_size = 1, 
										shuffle = True,
										drop_last=False)
	
		models_loc = "Trained_Models"
		Inference_dir = "Inference_Run"
		models = os.listdir(models_loc)

		for num, model in enumerate(models):
			choice = "[" + str(num) + "]    " + model
			print(choice)

		choice  = int(input("please input modelnum: "))

		modeldir = models_loc + "/"  + models[choice] + "/model.pt"
		modelname = models[choice]
		run_dir = Inference_dir + "/" + modelname
		print(run_dir)
		model_state = models_loc + "/"  + models[choice] + "/Savestate.txt"
		model_inf = []
		with open(model_state, 'r') as f:
			model_inf = [Line for Line in f]

		model_arch = model_inf[21].split(':')[1].strip() # 21 is the location of the model architecture in the savestate.txt

		if model_arch == "Generator_Unet1":
			Model = Generator_Unet1()

		elif model_arch == "UNet_ResNet34":
			Model = UNet_ResNet34()

		if os.path.isdir(run_dir):
				while True:
					choice = input("The directory already exists, would you like to replace it for new inference? [y/n]: ")
					if choice == "y":
						shutil.rmtree(run_dir)
						os.makedirs(run_dir)
						os.makedirs(run_dir + "/output")
						inference_run = Model_Inference(Model, imloader, Settings, modeldir, modelname, run_dir)
						inference_run.Inference_run()
						inference_run.CreateMetrics()
						break
					if choice == "n":
						break
		else:
			os.makedirs(run_dir)
			os.makedirs(run_dir + "/output")
			inference_run = Model_Inference(Model, imloader, Settings, modeldir, modelname, run_dir)
			inference_run.Inference_run()
			inference_run.CreateMetrics()
