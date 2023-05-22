from torch.utils.data import DataLoader
import torch
from Models.GAN_Model_1 import Generator_Unet1, Discriminator_1, UnetGenerator
from Models.GAN_REF_HEMIN import UNet_ResNet34
from Datasets.GAN_Dataset_1 import GAN_dataset
from Training_Framework import Model_Inference
import os
import shutil

if __name__ == '__main__':


		Current_model_list = ["Generator_Unet1", "UNet_ResNet34", "UnetGenerator"]
		model_ref_loc = [21, 23, 24]

		Machine_learning_dir = "/home/anders/Master-thesis-project/Machine_Learning" # should point to the Machine learning folder of the local directory
		Preprocess_dir = "/home/anders/Thesis_image_cache" # This can be set wherever you want. This is where the dataloader will create the image cache after converting the training images to tensor files.

		#Most of these settings don't matter. They're just here to make it easier to initalize the dataloader et.al
		#The fields that do matter are things like normalization, cuda, Do norm, Pin memory, preprocess storage, dataset loc, 
		Settings = {
				"epochs"                : 5,
				"batch_size"            : 1, # This must be 1 for inference!
				"L1__local_loss_weight" : 50, # Don't know how much higher than 100 is stable, 300 causes issues. Might be related to gradient calc. balooning.
				"L1_loss_weight"        : 50,
				"BoxSet"               : [3,10], # Low/high Boxsize of the error region. The value represents for length and with.
				"Loss_region_Box_mult"  : 3, # This is now static at 3, do not change!
				"lr"                    : 0.0002,
				"dataset_loc"           : Machine_learning_dir,
				"preprocess_storage"    : Preprocess_dir,
				"seed"                  : 172, # random training seed
				"num_workers"           : 1,
            "Data_mean"             : [0.3212, 0.3858, 0.2613],
            "Data_std"              : [0.2938, 0.2827, 0.2658],
            	"Do norm"               : True, #Normalization on or off 
				"shuffle"               : True,
				"Datasplit"             : 0.7,
				"device"                : "cuda",
				"ImageHW"               : 256,
				"RestoreModel"          : False,
				#No spaces in the model name, please use '_'
				"ModelTrainingName"     : "RESOURCE_TEST_DELETE_ME",
				"Drop_incomplete_batch" : True,
				"Num_training_samples"  : None, #Setting this to None makes the Dataloader use all available images.
				"Pin_memory"            : True
				}

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


		# Checking if model ran with normalization or not (legacy model support)
		was_norm = model_inf[14].split(':')[1].strip()
		print("was_norm:", was_norm)
		if (was_norm) == "True":
			print("Model detected with normalization")
			Settings["Do norm"] = True
		else:
			print("Model detected without normalization")
			Settings["Do norm"] = False


		Custom_dataset = GAN_dataset(Settings, preprocess=False)

		imloader = DataLoader(Custom_dataset,
										num_workers = 0,
										batch_size = 1, 
										shuffle = True,
										drop_last=False)

		# Deciding on model architecture
		model_arch = ""

		for model_name in Current_model_list:
			for loc in model_ref_loc:
				try:
					model_arch = model_inf[loc].split(':')[1].strip()
				except:
					pass

				if model_name == model_arch:
					print("found model arch!")
					print(model_arch)
					break
			else:
				continue
			break

		if model_arch == "Generator_Unet1":
			Model = Generator_Unet1()

		if model_arch == "UNet_ResNet34":
			Model = UNet_ResNet34()

		if model_arch == "UnetGenerator":
			Model = UnetGenerator(norm_layer=torch.nn.InstanceNorm2d)

		if os.path.isdir(run_dir):
				while True:
					choice = input("The directory already exists, would you like to replace it for new inference? [y/n]: ")
					if choice == "y":
						shutil.rmtree(run_dir)
						os.makedirs(run_dir)
						os.makedirs(run_dir + "/output")
						inference_run = Model_Inference(Model, imloader, Settings, modeldir, modelname=model_arch, run_dir=run_dir)
						inference_run.Inference_run()
						inference_run.CreateMetrics()
						break
					if choice == "n":
						break
		else:
			os.makedirs(run_dir)
			os.makedirs(run_dir + "/output")
			inference_run = Model_Inference(Model, imloader, Settings, modeldir, modelname=model_arch, run_dir=run_dir)
			inference_run.Inference_run()
			inference_run.CreateMetrics()
