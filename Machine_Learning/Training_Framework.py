import paramiko
import os
from datetime import datetime
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from ignite.metrics import PSNR as P_PSNR
from ignite.metrics import SSIM as P_SSIM
from Losses.Losses import LossFunctions
import time
import sys
from math import ceil, floor

#---------------- Helper functions ----------------------
def PIL_concatenate_h(arr):
    if len(arr) == 3:
        im1, im2, im3 = arr
        out = Image.new('RGB', (im1.width + im2.width + im3.width, im1.height))
        out.paste(im1, (0,0))
        out.paste(im2, (im1.width, 0))    
        out.paste(im3, (im1.width + im2.width, 0))
        return out
    else:
        im1,im2 = arr
        out = Image.new('RGB', (im1.width + im2.width, im1.height))
        out.paste(im1, (0,0))
        out.paste(im2, (im1.width, 0))    
        return out

def Readfile(dir):
    ReadList = []
    with open(dir, 'r') as reader:
        for line in reader:
            imagename = line[:-1]
            ReadList.append(imagename)
    return ReadList
#-------------- END helper functions --------------------



class FileSender():
    """
    This class sets up sending files between the local directory given (!Only expects no subfolders!) and the uio folder "Master_Thesis_Model_Directory/"
    It will make it easier to work with as the training cluster is not accessible to IP's outside the uio servers.

    *In the future maybe add some functionality to pull from the uio server to the local folder where this program is run.
    
    """
    def __init__(self):
        self.externaldir = "Master_Thesis_Model_Directory"
        print("setting up ssh and sftp")
        self.server = "login.uio.no"

        self.GetCredentials()
        self.cli = paramiko.SSHClient()
        self.cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.cli.connect(hostname=self.server, port=22, username=self.username, password=self.password)
        print("Succesfully connected to", self.server)
        self.ftr = self.cli.open_sftp()
        print("sftp open")
        self.local_Model_Directory = "Trained_Models"


    def GetCredentials(self):
        localpassdir = [
            "C:/Users/ander/Documents/Master-thesis-project/local_UiO_Password.txt",
            "G:/Master-thesis-project/local_UiO_Password.txt",
        ]
        for path in localpassdir:
            if os.path.isfile(path):
                with open(path, 'r') as f:
                    self.username = f.readline().strip()
                    self.password = f.readline().strip()
                    return
        self.username = input("input username: ")
        self.password = input("input password: ")  

    def send(self, directory):
        num = 0
        odir = None
        walker = tqdm(os.walk(directory), unit="file")
        for root, dirs, files in walker:
            walker.set_description(f"Sending {os.path.basename(root)} to server storage")
            foldername = os.path.basename(root)
            if num > 0 :
                foldername = odir + "/" + foldername
            odir = foldername
            self.ftr.mkdir(self.externaldir + "/" + foldername)
            for filename in files:
                file_external_path = self.externaldir + "/" + foldername + "/" + filename
                file_local_path = root + "/" + filename
                self.ftr.put(file_local_path ,file_external_path)
            print("finished sending directory", foldername)
            num += 1

    #Needs and update to support directories, but doesnt work for 2FA yet either.
    def pull(self, directory):
        dir_struct = tqdm(self.ftr.listdir(self.externaldir + "/" + directory), unit="file")
        os.makedirs(self.local_Model_Directory + "/" + directory)
        for filename in dir_struct:
            dir_struct.set_description(f"downloading folder {directory}/{filename}")
            file_external_path = self.externaldir + "/" + directory + "/" + filename
            file_local_path = self.local_Model_Directory + "/" + directory + "/" + filename
            self.ftr.get(file_external_path, file_local_path)
        print("retrieved ", directory, "from remote server")

    def get_remote_models(self):
        local_dir = "Trained_Models"
        local_list = os.listdir(local_dir)
        remote_list = self.ftr.listdir(self.externaldir)
        fetch_list = [dir for dir in remote_list if dir not in local_list]
        if len(fetch_list) == 0:
            print("found no new models")
        else:
            for folder in fetch_list:
                self.pull(folder)
            print("completed remote folder transfer")

    def push_local_models(self):
        local_dir = "Trained_Models"
        local_list = os.listdir(local_dir)
        remote_list = self.ftr.listdir(self.externaldir)
        fetch_list = [dir for dir in local_list if dir not in remote_list]
        if len(fetch_list) == 0:
            print("found no new models")
        else:
            for folder in fetch_list:
                folder = local_dir + "/" + folder
                self.send(folder)
            print("completed remote folder transfer")

    def close(self):
        self.ftr.close()
        self.cli.close()
        print("SSH and SFTP closed")

class NormalizeInverse(transforms.Normalize):
    """
    Reverses the normalization applied in the dataset. Class is borrowed from the official Pytorch forums and is not my creation
    link: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/20
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

class CalculateMetrics():

    def __init__(self, modelref, dataloderref, deviceref):
        self.model = modelref
        self.dataloader = dataloderref
        self.device = deviceref
        """
        Inputs:
            - How many iterations to run for calculating the metrics (This should be variable to speed up calculation during training.)
            - 
        
        """

    def get_defects(self, image, mask):
            num_defects = torch.sum(~mask).to(device=self.device)
            side = num_defects.divide(3).sqrt().floor().to(dtype=torch.int, device=self.device)
            Whole = side**2 * 3
            if num_defects == Whole:
                    real_B = torch.masked_select(image, ~mask).view(1,3,side,side)
            else:
                    real_B = torch.masked_select(image, ~mask)[:-(num_defects - Whole)].view(1,3,side,side)
            return real_B
        
    def ComputeMetrics(self, iterations, batch=16, std=False):
        """
        Should have toggle to only compute SSIM
        """

        total_len = iterations 
        psnr_calc = P_PSNR(data_range=1.0)
        ssim_calc = P_SSIM(data_range=1.0, kernel_size=(7,7))
        self.model.zero_grad()  
        with torch.no_grad():
            self.model.eval()
            total_images = total_len
            total_defects = total_len * batch
            PSNR_real_values = torch.zeros((total_images))
            PSNR_fake_values = torch.zeros((total_images))
            SSIM_real_values = torch.zeros((total_images))
            SSIM_fake_values = torch.zeros((total_images))

            PSNR_real_values_p = torch.zeros((total_defects))
            PSNR_fake_values_p = torch.zeros((total_defects))
            SSIM_real_values_p = torch.zeros((total_defects))
            SSIM_fake_values_p = torch.zeros((total_defects))
            lbar = tqdm(range(total_images), leave=False)
            for num in lbar:
                images, defect_images, defect_mask = next(iter(self.dataloader))
                lbar.set_description(f"Computing metrics {num+1}/{total_images} images")

                if num > (total_len - 1):
                    break
                
                #Load images and run inference 
                real_A = defect_images.to(self.device) #Defect
                real_B = images.to(self.device) #Target 
                fake_B, _ = self.model(real_A.clone())
                mask = defect_mask.to(self.device)

                if batch > 1:
                    for i in range(batch):
                        fake_B_local = self.get_defects(fake_B[i,:], mask[i,:])
                        real_B_local = self.get_defects(real_B[i,:], mask[i,:])
                        real_A_local = self.get_defects(real_A[i,:], mask[i,:])
                        psnr_calc.update((real_A_local, real_B_local))
                        PSNR_real_values_p[num] = psnr_calc.compute()
                        psnr_calc.reset()

                        psnr_calc.update((fake_B_local, real_B_local))
                        PSNR_fake_values_p[num] = psnr_calc.compute()
                        psnr_calc.reset()

                        ssim_calc.update((real_A_local, real_B_local))
                        SSIM_real_values_p[num] = ssim_calc.compute()
                        ssim_calc.reset()

                        ssim_calc.update((fake_B_local, real_B_local))
                        SSIM_fake_values_p[num] = ssim_calc.compute()
                        ssim_calc.reset()
                else:
                    fake_B_local = self.get_defects(fake_B, mask)
                    real_B_local = self.get_defects(real_B, mask)
                    real_A_local = self.get_defects(real_A, mask)

                    psnr_calc.update((real_A_local, real_B_local))
                    PSNR_real_values_p[num] = psnr_calc.compute()
                    psnr_calc.reset()

                    psnr_calc.update((fake_B_local, real_B_local))
                    PSNR_fake_values_p[num] = psnr_calc.compute()
                    psnr_calc.reset()

                    ssim_calc.update((real_A_local, real_B_local))
                    SSIM_real_values_p[num] = ssim_calc.compute()
                    ssim_calc.reset()

                    ssim_calc.update((fake_B_local, real_B_local))
                    SSIM_fake_values_p[num] = ssim_calc.compute()
                    ssim_calc.reset()

                psnr_calc.update((real_A, real_B))
                PSNR_real_values[num] = psnr_calc.compute()
                psnr_calc.reset()

                psnr_calc.update((fake_B, real_B))
                PSNR_fake_values[num] = psnr_calc.compute()
                psnr_calc.reset()

                ssim_calc.update((real_A, real_B))
                SSIM_real_values[num] = ssim_calc.compute()
                ssim_calc.reset()

                ssim_calc.update((fake_B, real_B))
                SSIM_fake_values[num] = ssim_calc.compute()
                ssim_calc.reset()

            self.model.train()
        if std:
            return [PSNR_real_values.mean(), PSNR_fake_values.mean(), PSNR_real_values_p.mean().mul(batch), PSNR_fake_values_p.mean().mul(batch), SSIM_real_values.mean(), SSIM_fake_values.mean(), SSIM_real_values_p.mean().mul(batch), SSIM_fake_values_p.mean().mul(batch), PSNR_real_values.std(), PSNR_fake_values.std(), PSNR_real_values_p.std(), PSNR_fake_values_p.std(), SSIM_real_values.std(), SSIM_fake_values.std(), SSIM_real_values_p.std(), SSIM_fake_values_p.std()]
    
        else:
            return [PSNR_real_values.mean(), PSNR_fake_values.mean(), PSNR_real_values_p.mean().mul(batch), PSNR_fake_values_p.mean().mul(batch), SSIM_real_values.mean(), SSIM_fake_values.mean(), SSIM_real_values_p.mean().mul(batch), SSIM_fake_values_p.mean().mul(batch)]
        
   

class DataCollectionClass():
    """
    TODO:
        - Need a function which collects for training and validation (combine these?)
        - Needs a saving function
        - Need a plotting function. 
    """
    
    def __init__(self, tr_len, vl_len, N_t, N_v, modeldir, Settings):
            # Required array size
            self.Modeldir = modeldir
            train_len = Settings["epochs"] * ceil(tr_len / N_t)
            val_len = Settings["epochs"] * ceil(vl_len / N_v)
            self.tr_iter = 0
            self.val_iter = 0
            #print("Sample rates for training and validation are:", N_t, N_v)
            #print("Sizes for the arrays are:", train_len, val_len)

            #Model Analytics
            self.Generator_loss_train = torch.zeros(train_len, requires_grad=False)
            self.Generator_pixel_loss_training = torch.zeros(train_len, requires_grad=False)
            self.Generator_local_pixel_loss_training = torch.zeros(train_len, requires_grad=False)
            self.Generator_DeepFeatureLoss_training = torch.zeros(train_len, requires_grad=False)    
            self.Generator_auto_loss_training = torch.zeros(train_len, requires_grad=False)

            self.Discriminator_loss_train = torch.zeros(train_len, requires_grad=False)
            self.Discriminator_accuracy_real_training_raw = torch.zeros(train_len, requires_grad=False)
            self.Discriminator_accuracy_fake_training_raw = torch.zeros(train_len, requires_grad=False)
            self.Discriminator_auto_loss_training = torch.zeros(train_len, requires_grad=False)

            #Val setup
            self.Generator_loss_validation = torch.zeros(val_len, requires_grad=False)
            self.Generator_pixel_loss_validation = torch.zeros(val_len, requires_grad=False)    
            self.Generator_local_pixel_loss_validation = torch.zeros(val_len, requires_grad=False)    
            self.Generator_DeepFeatureLoss_validation = torch.zeros(val_len, requires_grad=False)    
            self.Generator_auto_loss_validation = torch.zeros(val_len, requires_grad=False)

            self.Discriminator_loss_validation = torch.zeros(val_len, requires_grad=False)
            self.Discriminator_accuracy_real_validation_raw = torch.zeros(val_len, requires_grad=False)
            self.Discriminator_accuracy_fake_validation_raw = torch.zeros(val_len, requires_grad=False)    
            self.Discriminator_auto_loss_validation = torch.zeros(val_len, requires_grad=False)

            #Metric Analytics
            self.PSNR_Generated = torch.zeros(Settings["epochs"], requires_grad=False)
            self.PSNR_Generated_patch = torch.zeros(Settings["epochs"], requires_grad=False)
            self.SSIM_Generated = torch.zeros(Settings["epochs"], requires_grad=False)
            self.SSIM_Generated_patch = torch.zeros(Settings["epochs"], requires_grad=False)
            self.Model_Metric_Score = torch.zeros(Settings["epochs"], requires_grad=False)



    def Analytics_run(self, *args, val=False):
        batch, epoch, current_GEN_loss, current_DIS_loss, pixelloss, local_pixelloss, DeepFeatureloss_arr, Discrim_auto_loss, Generator_auto_loss, Discrim_acc_real_raw, Discrim_acc_fake_raw = args
        """
        current epoch needs to be the first argument, except when setting up training.  
        """
        
        if not val:
            self.Generator_loss_train[self.tr_iter] = current_GEN_loss.mean()
            self.Discriminator_loss_train[self.tr_iter] = current_DIS_loss.mean()
            self.Discriminator_accuracy_real_training_raw[self.tr_iter] = Discrim_acc_real_raw.mean()
            self.Discriminator_accuracy_fake_training_raw[self.tr_iter] = Discrim_acc_fake_raw.mean()
            self.Discriminator_auto_loss_training[self.tr_iter] = Discrim_auto_loss.mean()
            self.Generator_pixel_loss_training[self.tr_iter] = pixelloss.mean()
            self.Generator_local_pixel_loss_training[self.tr_iter] = local_pixelloss.mean()
            self.Generator_DeepFeatureLoss_training[self.tr_iter] = DeepFeatureloss_arr.mean()
            self.Generator_auto_loss_training[self.tr_iter] = Generator_auto_loss.mean()            
            self.tr_iter += 1
        else:
            self.Generator_loss_validation[self.val_iter] = current_GEN_loss.mean()
            self.Discriminator_loss_validation[self.val_iter] = current_DIS_loss.mean()
            self.Discriminator_accuracy_real_validation_raw[self.val_iter] = Discrim_acc_real_raw.mean()
            self.Discriminator_accuracy_fake_validation_raw[self.val_iter] = Discrim_acc_fake_raw.mean()
            self.Discriminator_auto_loss_validation[self.val_iter] = Discrim_auto_loss.mean()
            self.Generator_pixel_loss_validation[self.val_iter] = pixelloss.mean()
            self.Generator_local_pixel_loss_validation[self.val_iter] = local_pixelloss.mean()
            self.Generator_DeepFeatureLoss_validation[self.val_iter] = DeepFeatureloss_arr.mean()
            self.Generator_auto_loss_validation[self.val_iter] = Generator_auto_loss.mean()
            self.val_iter += 1

    def Metrics_run(self, args, epoch):
        PSNR_score, PSNR_score_p, SSIM_score, SSIM_score_p = args
        Score = PSNR_score + PSNR_score_p + SSIM_score + SSIM_score_p

        self.PSNR_Generated[epoch] = PSNR_score
        self.PSNR_Generated_patch[epoch] = PSNR_score_p
        self.SSIM_Generated[epoch] = SSIM_score
        self.SSIM_Generated_patch[epoch] = SSIM_score_p
        self.Model_Metric_Score[epoch] = Score

    def GetCurrentLoss(self, val=False):
        if not val:
            return self.Generator_loss_train[self.tr_iter-1], self.Discriminator_loss_train[self.tr_iter-1]
        else:
            return self.Generator_loss_validation[self.val_iter-1], self.Discriminator_loss_validation[self.val_iter-1]
        

    def Save_Analytics(self):
        torch.save((self.Generator_loss_validation,
                                self.Discriminator_loss_validation,
                                self.Discriminator_accuracy_real_validation_raw,
                                self.Discriminator_accuracy_fake_validation_raw,
                                self.Discriminator_auto_loss_training,
                                self.Generator_pixel_loss_validation,
                                self.Generator_local_pixel_loss_validation,
                                self.Generator_DeepFeatureLoss_validation,
                                self.Generator_DeepFeatureLoss_training,
                                self.Generator_loss_train,
                                self.Generator_pixel_loss_training,
                                self.Generator_local_pixel_loss_training,
                                self.Discriminator_loss_train,
                                self.Discriminator_accuracy_real_training_raw,
                                self.Discriminator_accuracy_fake_training_raw,
                                self.Discriminator_auto_loss_validation,
                                self.Generator_auto_loss_validation,
                                self.PSNR_Generated,
                                self.PSNR_Generated_patch,
                                self.SSIM_Generated,
                                self.SSIM_Generated_patch,
                                self.Model_Metric_Score,
                                ), self.Modeldir + '/Analytics.pt')


    def MakeSaveGraph(self, datas, xlabel, ylabel, title):
        for axis, legend in datas:
            xaxis = torch.arange(axis.size(0))
            plt.plot(xaxis, axis, label=legend)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.savefig(self.Modeldir + "/" + title + ".png")
        plt.clf()

    def Create_graphs(self):
        #Currently not making the joined Discriminator / Generator loss, nor the Discriminator real/fake raw pred graphs
        self.MakeSaveGraph([[self.Generator_loss_train, "Training"],[ self.Generator_loss_validation, "Validation"]], "samples", "loss (WGAN)", "Generator Loss")
        self.MakeSaveGraph([[self.Discriminator_loss_train, "Training"],[ self.Discriminator_loss_validation, "Validation"]], "samples", "loss (WGAN)", "Discriminator Loss ")
        self.MakeSaveGraph([[self.Generator_DeepFeatureLoss_training, "Training"],[ self.Generator_DeepFeatureLoss_validation, "Validation"]], "samples", "loss (WGAN)", "Generator Latent Feature Loss")
        self.MakeSaveGraph([[self.Discriminator_auto_loss_training, "Training"],[ self.Discriminator_auto_loss_validation, "Validation"]], "samples", "loss (WGAN)", "Discriminator Autoencoder Loss")
        self.MakeSaveGraph([[self.Generator_auto_loss_training, "Training"],[ self.Generator_auto_loss_validation, "Validation"]], "samples", "loss (WGAN)", "Generator Autoencoder Loss")
        self.MakeSaveGraph([[self.Generator_pixel_loss_training, "Training"],[ self.Generator_pixel_loss_validation, "Validation"]], "samples", "loss (L1)", "Generator Pixel Loss")
        self.MakeSaveGraph([[self.Generator_local_pixel_loss_training, "Training"],[ self.Generator_local_pixel_loss_validation, "Validation"]], "samples", "loss (MSE)", "Generator Local Pixel Loss")
        self.MakeSaveGraph([[self.PSNR_Generated, "Global"],[ self.PSNR_Generated_patch, "Defect"]], "epochs", "PSNR (dB)", "PSNR per epoch")
        self.MakeSaveGraph([[self.SSIM_Generated, "Global"],[ self.SSIM_Generated_patch, "Defect"]], "epochs", "SSIM (%)", "SSIM per epoch")
        self.MakeSaveGraph([[self.Model_Metric_Score, "Model Score"]], "epochs", "Score [PSNR + SSIM global/local]", "Model score")

class Training_Framework():
    """
    Framework for training the different networks, should not inherit or mess with pytorch itself, but instead passes the model by assignment to make training more like 
    Legos and less like a novel.
    """

    def __init__(self, Settings, Autoencoder, Generator, G_opt, D_opt, Discriminator, train_loader, val_loader):
        torch.manual_seed(Settings["seed"])
        np.random.seed(Settings["seed"])
        self.Settings = Settings
        self.Autoencoder = Autoencoder
        if self.Autoencoder is not None:
            self.Autoencoder.eval()
        self.Generator = Generator
        self.Generator_optimizer = G_opt
        self.Discriminator = Discriminator
        self.image_transform = transforms.ToPILImage()
        self.Discriminator_optimizer = D_opt
        self.device = self.Settings["device"]
        self.Generator_loss = 0
        self.n_crit = self.Settings["n_crit"]
        self.lambda_gp = self.Settings["lambda_gp"]
        self.Discriminator_loss = 0
        self.Reverse_Normalization = NormalizeInverse(self.Settings["Data_mean"], self.Settings["Data_std"])
        #self.patch = torch.tensor((1, self.Settings["ImageHW"] // 2 ** 4, self.Settings["ImageHW"] // 2 ** 4), device=self.device)
        self.transmit = False # No reason to do file transfer in the future
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.InpainterRun = False # inpainter flag
        self.AutoencoderRun = False # Autoencoder flag
        #Initialize loss functions
        self.losses = LossFunctions(self.device, Discriminator, Settings)

        if Settings["Loss"] == "Hinge_loss":
            self.Discriminator_loss                 = self.losses.Hinge_loss_Discriminator
            self.Generator_loss                     = self.losses.Hinge_loss_Generator
            self.Generator_pixelloss                = self.losses.Generator_Pixelloss
            self.Generator_autoencoder_pixelloss    = self.losses.Generator_Autoencoder_Pixelloss
            self.Generator_Deep_Feature_Loss        = self.losses.Latent_Feature_Criterion
        if Settings["Loss"] == "WGAN":
            self.Discriminator_loss                 = self.losses.WGAN_Discriminator
            self.Generator_loss                     = self.losses.WGAN_Generator
            self.Generator_pixelloss                = self.losses.Generator_Pixelloss
            self.Generator_autoencoder_pixelloss    = self.losses.Generator_Autoencoder_Pixelloss
            self.Generator_Deep_Feature_Loss        = self.losses.Latent_Feature_Criterion
        if Settings["Loss"] == "CGAN":
            self.Discriminator_loss                 = self.losses.CGAN_Discriminator
            self.Generator_loss                     = self.losses.CGAN_Generator
            self.Generator_pixelloss                = self.losses.Generator_Pixelloss
            self.Generator_autoencoder_pixelloss    = self.losses.Generator_Autoencoder_Pixelloss
            self.Generator_Deep_Feature_Loss        = self.losses.Latent_Feature_Criterion
        if Settings["Loss"] == "WGANGP":
            self.Discriminator_loss                 = self.losses.WGANGP_Discriminator
            self.Generator_loss                     = self.losses.WGAN_Generator
            self.Generator_pixelloss                = self.losses.Generator_Pixelloss
            self.Generator_autoencoder_pixelloss    = self.losses.Generator_Autoencoder_Pixelloss
            self.Generator_Deep_Feature_Loss        = self.losses.Latent_Feature_Criterion


        if Settings["Objective"] == "Inpainting":
            self.InpainterRun = True # inpainter flag
            self.Discriminator_updater              = self.Discriminator_Inpainting_updater
            self.Generator_updater                  = self.Generator_Inpainting_updater
        if Settings["Objective"] == "AutoEncoder":
            self.AutoencoderRun = True # Autoencoder flag
            self.Discriminator_updater              = self.Discriminator_Autoencoder_updater
            self.Generator_updater                  = self.Generator_Autoencoder_updater         
        if Settings["Objective"] == "DualEncoder":
            self.InpainterRun = True # inpainter flag
            self.AutoencoderRun = True # Autoencoder flag
            self.Discriminator_updater              = self.Discriminator_DualEncoder_updater
            self.Generator_updater                  = self.Generator_DualEncoder_updater
        if Settings["Objective"] == "InpainterWithLatentAutoencoder":
            self.InpainterRun = True # inpainter flag
            self.AutoencoderRun = True # Autoencoder flag
            self.Discriminator_updater              = self.Discriminator_Inpainting_updater
            self.Generator_updater                  = self.Generator_InpaintLatent_updater
        if Settings["Objective"] == "NoGAN":
            self.InpainterRun = True # inpainter flag
            self.Discriminator_updater              = self.Discriminator_NoGAN
            self.Generator_updater                  = self.Generator_NoGAN

        # Set the working Directory
        if not self.Settings["dataset_loc"]:
            self.workingdir = os.getcwd()
        else:
            self.workingdir = self.Settings["dataset_loc"]

        #Create the directory of the model (Look back at this for batch training.)
        times = str(datetime.now())
        stamp = times[:-16] + "_" + times[11:-7].replace(":", "-")
        if self.Settings["ModelTrainingName"] is not None:
            self.Modeldir = self.workingdir +  "/Trained_Models/" + self.Settings["ModelTrainingName"] + " (" + stamp +")"
        else:
            self.Modeldir = self.workingdir + "/Trained_Models/" + "GAN_Model" + " " + stamp
            
        os.makedirs(self.Modeldir)

        #Creating folders for image output during training.
        self.modeltraining_output               = self.Modeldir + "/training_output"
        os.makedirs(self.modeltraining_output)

        #Initializing Data collection
        self.N_training_sample_rate = Settings["Training_sample_rate"] # User defined sample rate
        self.N_validation_sample_rate = ceil((len(val_loader) * self.N_training_sample_rate) / len(train_loader)) # Validation sample rate defined by the size of the training sample rate. This is so both arrays are the same size at the end of training. 
        self.Collector = DataCollectionClass(len(train_loader), len(val_loader), self.N_training_sample_rate, self.N_validation_sample_rate, self.Modeldir, Settings)
        self.train_start = time.time()

        #Initializing metrics and score class
        self.Evaluator = CalculateMetrics(self.Generator, self.val_loader, self.device)
        #Putting the model score array here temporarily
        self.ModelScores = torch.zeros(Settings["epochs"])

    def Save_Model(self, epoch):
        with torch.no_grad():
            self.Generator.eval()
            self.Discriminator.eval()
            results = self.Evaluator.ComputeMetrics(6) # desired 100 images -> 100/batch(16) = 6.25
            self.Collector.Metrics_run([results[1], results[3], results[5]*100, results[7]*100], epoch)
            Score = results[1] + results[3] + results[5]*100 + results[7]*100
            if epoch == 0 or Score > self.ModelScores.max():
                print("saved model, score:", Score)
                torch.save(self.Generator.state_dict(), str( self.Modeldir + "/model.pt"))
                torch.save(self.Discriminator.state_dict(), str( self.Modeldir + "/dis_model.pt"))
        self.Generator.train()
        self.Discriminator.train()
        self.ModelScores[epoch] = Score

    def SaveState(self):
        training_time_seconds = time.time() - self.train_start
        minutes, training_time_seconds = divmod(training_time_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        PSettings = list(self.Settings)
        stateloc = self.Modeldir + "/Savestate.txt"
        with open(stateloc, 'w') as f:
            for param in PSettings:
                f.write(param +  ": " + str(self.Settings[param]) + "\n")
            f.write("Generator model: " + self.Generator.name + "\n")
            f.write("Discriminator model: " + self.Discriminator.name + "\n")
            f.write("Total training time: " + str(hours) + " hours " + str(minutes) + " minutes \n")
            f.write("Chosen loss type: " + self.Settings["Loss"] + "\n")
            try:
                f.write("Loss function for Latent loss: " + self.Generator_Deep_Feature_Loss.__name__ + "\n")
            except AttributeError:
                f.write("Loss function for Latent loss: " + type(self.Generator_Deep_Feature_Loss).__name__ + "\n")
            f.write("Loss function for Discriminator loss: " + self.Discriminator_loss.__name__ + "\n")
            f.write("Loss function for Generator loss: " + self.Generator_loss.__name__ + "\n")
            f.write("Loss function for Generator pixel loss: " + self.Generator_pixelloss.__name__ + "\n")
            f.write("Generator updater function: " + self.Generator_updater.__name__ + "\n")
            f.write("Number of training samples:" + str(len(self.train_loader) * 16) + "\n")
            f.write("Number of validation samples:" + str(len(self.val_loader) * 16) + "\n")
            f.write("Discriminator updater function: " + self.Discriminator_updater.__name__ + "\n")
            f.write("Loss function for Generator auto encoder pixel loss: " + self.Generator_autoencoder_pixelloss.__name__ + "\n")
            f.write("Internal Loss criterion for global pixel loss: " + type(self.losses.pixelwise_loss).__name__ + "\n")
            f.write("Internal Loss criterion for local pixel loss: " + type(self.losses.pixelwise_local_loss).__name__ + "\n")
            f.write("Internal Loss criterion for latent loss  (if not WGAN): " + type(self.losses.Latent_Feature_Criterion).__name__ + "\n")
            f.write("Final model score: " + str(self.ModelScores[-1]) + "\n")


    def Generator_InpaintLatent_updater(self, val=False):
        self.Generator.zero_grad()    
        # Generator GAN loss
        # BB is autoencoder output 
        # BA is inpainting output

        fake_BA = torch.cat((self.fake_BA, self.real_B), 1)     
        predicted_fake_BA = self.Discriminator(fake_BA)

        loss_GAN_BA = self.Generator_loss(predicted_fake_BA)

        #Pixelwise loss
        loss_pixel_BA, local_pixelloss_BA       = self.Generator_pixelloss(self.fake_BA, self.real_B, self.mask)

        total_pixelloss_BA                      = loss_pixel_BA + local_pixelloss_BA * self.Settings["L1__local_loss_weight"]

        #Latent Feature loss
        _, self.Latent_BB = self.Autoencoder(self.real_B) # Autoencoder run
        LatentLoss = self.Generator_Deep_Feature_Loss(self.Latent_BA, self.Latent_BB) * self.Settings["Latent_loss_weight"]    
        loss_GAN_BB = total_pixelloss_BB = torch.zeros(1)
        #Total loss
        Total_loss_Generator = loss_GAN_BA + LatentLoss + total_pixelloss_BA
        
        if not val:
            Total_loss_Generator.backward()
            self.Generator_optimizer.step()  

        return loss_GAN_BA.detach(), loss_GAN_BB.detach(), loss_pixel_BA.detach(), local_pixelloss_BA.detach(), LatentLoss.detach()   

    def Generator_DualEncoder_updater(self, val=False):
        self.Generator.zero_grad()    
        # Generator GAN loss
        fake_BB = torch.cat((self.fake_BB, self.real_B), 1)     
        predicted_fake_BB = self.Discriminator(fake_BB)
        fake_BA = torch.cat((self.fake_BA, self.real_B), 1)     
        predicted_fake_BA = self.Discriminator(fake_BA)

        loss_GAN_BA = self.Generator_loss(predicted_fake_BA)

        loss_GAN_BB = self.Generator_loss(predicted_fake_BB)
        
        #Pixelwise loss
        loss_pixel_BB, local_pixelloss_BB       = self.Generator_pixelloss(self.fake_BB, self.real_B, self.mask)
        loss_pixel_BA, local_pixelloss_BA       = self.Generator_pixelloss(self.fake_BA, self.real_B, self.mask)

        total_pixelloss_BA                      = loss_pixel_BA + local_pixelloss_BA * self.Settings["L1__local_loss_weight"]
        total_pixelloss_BB                      = loss_pixel_BB * self.Settings["L1_loss_weight"] + local_pixelloss_BB

        #Latent Feature loss
        LatentLoss = self.Generator_Deep_Feature_Loss(self.Latent_BA, self.Latent_BB) * self.Settings["Latent_loss_weight"]    

        #Total loss
        Total_loss_Generator = loss_GAN_BB + loss_GAN_BA + LatentLoss + total_pixelloss_BA + total_pixelloss_BB #+ LatentLoss
        
        if not val:
            Total_loss_Generator.backward()
            self.Generator_optimizer.step()  

        return loss_GAN_BA.detach(), loss_GAN_BB.detach(), loss_pixel_BA.detach(), local_pixelloss_BA.detach(), LatentLoss.detach()        
    

    def Generator_Autoencoder_updater(self, val=False):  
        self.Generator.zero_grad()    
        # Generator GAN loss
        fake_BB = torch.cat((self.fake_BB, self.real_B), 1)     
        predicted_fake_BB = self.Discriminator(fake_BB)


        loss_GAN_BB = self.Generator_loss(predicted_fake_BB)
        
        #Pixelwise loss
        loss_pixel =  self.Generator_autoencoder_pixelloss(self.fake_BB, self.real_B)#self.Generator_pixelloss(self.fake_BB, self.real_B, self.mask)

        total_pixelloss_BB = loss_pixel * self.Settings["L1_loss_weight"]

        #Latent Feature loss
        #LatentLoss = self.Generator_Deep_Feature_Loss(self.Latent_BA, self.Latent_BB)        
        LatentLoss = torch.zeros(1)
        local_pixelloss = torch.zeros(1)
        

        #Total loss
        Total_loss_Generator = loss_GAN_BB  + total_pixelloss_BB #+ LatentLoss
        
        if not val:
            Total_loss_Generator.backward()
            self.Generator_optimizer.step()  
        
        # return loss_GAN_BA.detach(), loss_GAN_BB.detach(), loss_pixel_BA.detach(), local_pixelloss_BA.detach(), LatentLoss.detach()        
        return loss_GAN_BB.detach(), LatentLoss.detach(), loss_pixel.detach(), local_pixelloss.detach(), LatentLoss.detach()
    
    def Generator_Inpainting_updater(self, val=False): 
        self.Generator.zero_grad()       
        # Generator GAN loss
        fake_BA = torch.cat((self.fake_BA, self.real_B), 1)     
        predicted_fake_BA = self.Discriminator(fake_BA)

        loss_GAN_BA = self.Generator_loss(predicted_fake_BA)

        #Pixelwise loss
        loss_pixel, local_pixelloss =  self.Generator_pixelloss(self.fake_BA, self.real_B, self.mask)

        LatentLoss = torch.zeros(1)
        #Latent Feature loss # only viable for training with dual encoder scheme's 
        #LatentLoss = self.Generator_Deep_Feature_Loss(self.Latent_BA, self.Latent_BB)    

        #Total loss
        Generator_Inpainting_loss = loss_GAN_BA + loss_pixel * self.Settings["L1_loss_weight"] + local_pixelloss * self.Settings["L1__local_loss_weight"]
        
        Total_loss_Generator = Generator_Inpainting_loss
        if not val:
            Total_loss_Generator.backward()
            self.Generator_optimizer.step()
   
        # return loss_GAN_BA.detach(), loss_GAN_BB.detach(), loss_pixel_BA.detach(), local_pixelloss_BA.detach(), LatentLoss.detach()        
        return loss_GAN_BA.detach(), LatentLoss.detach(), loss_pixel.detach(), local_pixelloss.detach(), LatentLoss.detach()  

    def Generator_NoGAN(self, val=False): 
        self.Generator.zero_grad()       
        #Pixelwise loss
        loss_pixel, local_pixelloss =  self.Generator_pixelloss(self.fake_BA, self.real_B, self.mask)

        LatentLoss = torch.zeros(1)
        loss_GAN_BA = torch.zeros(1) 

        #Total loss
        Generator_Inpainting_loss =  loss_pixel * self.Settings["L1_loss_weight"] + local_pixelloss * self.Settings["L1__local_loss_weight"]
        
        Total_loss_Generator = Generator_Inpainting_loss
        if not val:
            Total_loss_Generator.backward()
            self.Generator_optimizer.step()
   
        # return loss_GAN_BA.detach(), loss_GAN_BB.detach(), loss_pixel_BA.detach(), local_pixelloss_BA.detach(), LatentLoss.detach()        
        return loss_GAN_BA.detach(), LatentLoss.detach(), loss_pixel.detach(), local_pixelloss.detach(), LatentLoss.detach()  

    def Discriminator_DualEncoder_updater(self, val=False):
        self.Discriminator.zero_grad()
        #Get Critique scores
        fake_BB = torch.cat((self.fake_BB, self.real_B), 1)
        pred_fake_BB = self.Discriminator(fake_BB.detach())

        fake_BA = torch.cat((self.fake_BA, self.real_A), 1) 
        real_AB = torch.cat((self.real_A, self.real_B), 1)     
        pred_fake_BA = self.Discriminator(fake_BA.detach())
        pred_real_AB = self.Discriminator(real_AB)
        
        Discriminator_auto_loss = self.Discriminator_loss(pred_fake_BB, pred_real_AB)
        Discriminator_inpaint_loss = self.Discriminator_loss(pred_fake_BA, pred_real_AB)

        Discriminator_loss = Discriminator_auto_loss + Discriminator_inpaint_loss

        if not val:
            Discriminator_loss.backward(retain_graph=True)
            self.Discriminator_optimizer.step()

        return Discriminator_inpaint_loss.detach(), Discriminator_auto_loss.detach(), pred_real_AB.detach(), pred_fake_BB.detach()            


    def Discriminator_Autoencoder_updater(self, val=False):
        self.Discriminator.zero_grad()
        #Get Critique scores
        fake_BB = torch.cat((self.fake_BB, self.real_B), 1)
        real_AB = torch.cat((self.real_B, self.real_B), 1)     #this used to compare real_A with real_B, but it shouldn't?
        pred_real_AB = self.Discriminator(real_AB)
        pred_fake_BB = self.Discriminator(fake_BB.detach())

        Discriminator_auto_loss = self.Discriminator_loss(pred_fake_BB, pred_real_AB)
        Discriminator_inpaint_loss = torch.zeros(1)


        if not val:
            Discriminator_auto_loss.backward(retain_graph=True)
            self.Discriminator_optimizer.step()
        return Discriminator_inpaint_loss.detach(), Discriminator_auto_loss.detach(), pred_real_AB.detach(), pred_fake_BB.detach()

    def Discriminator_NoGAN(self, val=False):
        dummy_return_val = torch.zeros(1, requires_grad=False, device=self.device)  
        return dummy_return_val, dummy_return_val, dummy_return_val, dummy_return_val
    
    def Discriminator_Inpainting_updater(self, val=False):
        self.Discriminator.zero_grad()

        #Get Critique scores
        fake_BA = torch.cat((self.fake_BA, self.real_B), 1)   
        real_AB = torch.cat((self.real_A, self.real_B), 1)     
        pred_fake_BA = self.Discriminator(fake_BA.detach())
        pred_real_AB = self.Discriminator(real_AB)

        #Calculate loss # loss_real = - torch.mean(real_pred) loss_fake = torch.mean(fake_pred)
        Discriminator_inpaint_loss = self.Discriminator_loss(pred_fake_BA, pred_real_AB)
        Discriminator_auto_loss = torch.zeros(1)

        if not val:
            Discriminator_inpaint_loss.backward(retain_graph=True)
            self.Discriminator_optimizer.step()

        return Discriminator_inpaint_loss.detach(), Discriminator_auto_loss.detach(), pred_real_AB.detach(), pred_fake_BA.detach()

    
    def FromTorchTraining(self, image):
        #Returns a trainable tensor back into a visual image.
        if self.Settings["Do norm"]:
            return self.Reverse_Normalization(image).permute(1,2,0).mul_(255).clamp(0,255).to("cpu", torch.uint8).numpy()
        else:
            return image.permute(1,2,0).mul_(255).clamp(0,255).to("cpu", torch.uint8).numpy()



    def Generate_validation_images(self, epoch):
        self.Generator.eval()

        if self.real_A.size(0) > 1:
            real_im = self.real_A[0,:,:,:] # Change B for A if working with inpainter, else leave it on real_B for autoencoder images
            mask    = self.mask[0,:,:,:]
        else:
            real_im = self.real_A
            mask    = self.mask

        with torch.no_grad():
            if self.real_A.size(0) > 1:
                fake_B, _ = self.Generator(real_im.unsqueeze(0))
                im = Image.fromarray(self.FromTorchTraining(fake_B.squeeze(0)))
                co = Image.fromarray(self.FromTorchTraining(real_im))
                ma = Image.fromarray(self.FromTorchTraining(mask.squeeze(0).int()))
            else:
                fake_B, _ = self.Generator(real_im)
                im = Image.fromarray(self.FromTorchTraining(fake_B.squeeze(0)))
                co = Image.fromarray(self.FromTorchTraining(real_im.squeeze(0)))
                ma = Image.fromarray(self.FromTorchTraining(mask.squeeze(0).int()))
            PIL_concatenate_h([ma, co, im]).save(self.modeltraining_output + "/" + "Image_" + str(epoch) + ".jpg", "JPEG")

        self.Generator.train()

    def validation_run(self, epoch):
        with torch.no_grad():

            if self.Settings["batch_size"] == 1:
                val_unit = "image(s)"
            else:
                val_unit = "batch(s)"

            #Turn off propagation
            self.Generator.eval()
            self.Generator.requires_grad=False
            self.Discriminator.eval()
            self.Discriminator.requires_grad=False
            last_GEN_loss, last_DIS_loss = self.Collector.GetCurrentLoss(val=True)       
            with tqdm(self.val_loader, unit=val_unit, leave=False) as tepoch:
                for num, data in enumerate(tepoch):
                    images, defect_images, mask = data
                    if epoch == 0:
                        tepoch.set_description(f"Validation run on Epoch {epoch}/{self.Settings['epochs']}")
                    elif epoch > 0:
                        tepoch.set_description(f"Validation Gen_loss {last_GEN_loss:.4f} Disc_loss {last_DIS_loss:.4f}")

                    self.real_A = defect_images.to(self.device) #Defect
                    self.real_B = images.to(self.device) #Target
                    if self.InpainterRun:
                        self.fake_BA, self.Latent_BA = self.Generator(self.real_A) # Inpainting run
                    if self.AutoencoderRun:
                        self.fake_BB, self.Latent_BB = self.Generator(self.real_B) # Autoencoder run
                    self.mask = mask.to(self.device) # local loss coordinates
                    DIS_loss, DIS_AutoEncoder_loss, predicted_real, predicted_fake = self.Discriminator_updater(val=True)
                    GEN_loss, GEN_AutoEncoder_loss, loss_pixel, loss_pixel_local, DeepFeatureLoss = self.Generator_updater(val=True)

                    #Analytics
                    if num == 0 or num % self.N_validation_sample_rate == 0:
                        self.Collector.Analytics_run(num, epoch, GEN_loss, DIS_loss, loss_pixel, loss_pixel_local, DeepFeatureLoss, DIS_AutoEncoder_loss, GEN_AutoEncoder_loss, torch.mean(predicted_real), torch.mean(predicted_fake), val=True)

            #Turn on propagation
            self.Generator.train()
            self.Generator.requires_grad=True
            self.Discriminator.train()
            self.Discriminator.requires_grad=True

            self.Generate_validation_images(epoch)

    def Trainer(self):
            self.fake_BA = None
            self.Latent_BA = None
            self.fake_BB = None 
            self.Latent_BB = None 
            epochs = tqdm(range(self.Settings["epochs"]), unit="epoch")
            for epoch in epochs:
                epochs.set_description(f"Training the model on epoch {epoch}")

                if self.Settings["batch_size"] == 1:
                    tepoch = tqdm(self.train_loader, unit='image(s)', leave=False)
                else:
                    tepoch = tqdm(self.train_loader, unit='batch(s)', leave=False)

                last_GEN_loss, last_DIS_loss = self.Collector.GetCurrentLoss()       
                for num, data in enumerate(tepoch):
                    images, defect_images, mask = data
                    if epoch == 0:
                        tepoch.set_description(f"Training on Epoch {epoch}/{self.Settings['epochs']}")
                    elif epoch > 0:
                        tepoch.set_description(f"Training Gen_loss {last_GEN_loss:.4f} Disc_loss {last_DIS_loss:.4f}")
                        
                    
                    self.real_A = defect_images.to(self.device) #Defect
                    self.real_B = images.to(self.device) #Target
                    if self.InpainterRun:
                        self.fake_BA, self.Latent_BA = self.Generator(self.real_A) # Inpainting run
                    if self.AutoencoderRun:
                        self.fake_BB, self.Latent_BB = self.Generator(self.real_B) # Autoencoder run

                    self.mask = mask.to(self.device) # local loss coordinates
                    # Discriminator return Discriminator_loss.detach(), autoencoder_loss.detach(), pred_real_AB.detach(), pred_fake_BA.detach()
                    #Generator    return loss_GAN_BA.detach(), loss_GAN_BB.detach(), loss_pixel.detach(), local_pixelloss.detach(), LatentLoss.detach()
                    DIS_loss, DIS_AutoEncoder_loss, predicted_real, predicted_fake = self.Discriminator_updater()
                    if num % self.n_crit == 0:
                        GEN_loss, GEN_AutoEncoder_loss, loss_pixel, local_loss_pixel, DeepFeatureloss = self.Generator_updater()

                    if num == 0 or num % self.N_training_sample_rate == 0:
                        self.Collector.Analytics_run(num, epoch, GEN_loss, DIS_loss, loss_pixel, local_loss_pixel, DeepFeatureloss, DIS_AutoEncoder_loss, GEN_AutoEncoder_loss, torch.mean(predicted_real), torch.mean(predicted_fake))   
                #Save per epoch
                self.validation_run(epoch)
                self.Save_Model(epoch)
            #Save Analytics to file and create images from analytics    
            self.Collector.Save_Analytics()
            self.Collector.Create_graphs()
            self.SaveState()
            #Create graph of the model scores over the epochs
            

            #Something fishy about metric calc at the end of training. Using the inference.py script until further notice
            #Metrics = Model_Inference(self.Generator, val_loader, self.Settings, self.Modeldir, training=True)
            #Metrics.CreateMetrics()
            #self.CreateMetrics(metric_loader) # Hook this up to the metric loader from the inference class instead
            if self.transmit: # Send finished model to server storage
                print("Sending files")
                self.transmitter.send(self.Modeldir)
                self.transmitter.close()


class Model_Inference():
    """
    Class to do inference and get inference data from the model.
    Made to only do inference on the Generator part of the GAN network.

    Input:
        - Modelref: This is the vanilla model class itself, from the Models/ dir
        - Modeldir: This is the location of the trained model.pt file.
    """
    def __init__(self, modelref, dataloader, Settings, modeldir, modelname=None, run_dir=None, device="cuda", training=False):
        self.Settings = Settings
        self.model = modelref
        self.device = device
        self.dataloader = dataloader
        self.training = training
        self.modeldir = modeldir
        self.modelname = modelname
        self.Reverse_Normalization = NormalizeInverse(self.Settings["Data_mean"], self.Settings["Data_std"])
        self.run_dir = run_dir
        self.BoxSet = self.Settings["BoxSet"]
        self.metric = CalculateMetrics(self.model, self.dataloader, self.device)
        if input("Autoencoder [y/n]?: "):
            self.autoencoder = True
        else:
            self.autoencoder = False

        if not training:
            self.RestoreModel()
        


    def RestoreModel(self):
        self.model.load_state_dict(torch.load(self.modeldir, map_location=torch.device(self.device)))
        self.model.to(self.device)
        print("Succesfully loaded", self.modelname, "model")

    def FromTorchTraining(self, image):
        #Returns a trainable tensor back into a visual image.
        if self.Settings["Do norm"]:
            return self.Reverse_Normalization(image).permute(1,2,0).mul_(255).clamp(0,255).to("cpu", torch.uint8).numpy()
        else:
            return image.permute(1,2,0).mul_(255).clamp(0,255).to("cpu", torch.uint8).numpy()
        
    def Inference_run(self, runs=30):
        """
        Does an inference run on the Model for three images
        """
        self.model.eval()
        loader = tqdm(range(runs))
        with torch.no_grad():
            for run in loader:
                loader.set_description(f"Running {run+1}/{runs} images completed")
                if self.autoencoder:
                    gt_image ,_ , _ = next(iter(self.dataloader))
                    real_B = gt_image.to(self.device)
                    fake_B, _ = self.model(real_B.clone())
                    im = Image.fromarray(self.FromTorchTraining(fake_B.squeeze(0)))
                    co = Image.fromarray(self.FromTorchTraining(real_B.squeeze(0)))
                    PIL_concatenate_h([co, im]).save(self.run_dir + "/output/image_" + str(run) + ".jpg")                
                else:
                    real_image ,defect_image , mask = next(iter(self.dataloader))
                    real_A = defect_image.to(device=self.device)
                    fake_B, _ = self.model(real_A.clone())
                    real_B = real_image.to(device=self.device)
                    defect_mask = mask.to(device=self.device)

                    Ground_truth_area = Image.fromarray(self.FromTorchTraining(self.metric.get_defects(real_B, defect_mask).squeeze(0)))
                    Generated_area = Image.fromarray(self.FromTorchTraining(self.metric.get_defects(fake_B, defect_mask).squeeze(0)))
                    Defect_area = Image.fromarray(self.FromTorchTraining(self.metric.get_defects(real_A, defect_mask).squeeze(0)))

                    Ground_truth_area.save(self.run_dir + "/output/image_" + str(run) + "_ground_truth_area.jpg")
                    Generated_area.save(self.run_dir + "/output/image_" + str(run) + "_generated_area.jpg")
                    Defect_area.save(self.run_dir + "/output/image_" + str(run) + "_defect_area.jpg")

                    im = Image.fromarray(self.FromTorchTraining(fake_B.squeeze(0)))
                    co = Image.fromarray(self.FromTorchTraining(real_A.squeeze(0)))
                    ma = Image.fromarray(self.FromTorchTraining(defect_mask.squeeze(0).int()))
                    PIL_concatenate_h([ma, co, im]).save(self.run_dir + "/output/image_" + str(run) + ".jpg")

        print("Done!")
        print("All results saved to:")
        print(self.run_dir)

    def CenteringAlgorithm(self, BoxSize, BoundingBox, Y, X):
        """
        Returns a larger bounding box centered on the defect block
        """

        x = torch.round(X + 0.5*BoxSize - 0.5*BoundingBox).to(torch.int).clamp(0, 256 - BoundingBox)
        y = torch.round(Y + 0.5*BoxSize - 0.5*BoundingBox).to(torch.int).clamp(0, 256 - BoundingBox)

        return y,x

    def CreateMetrics(self, total_len=500):
        results = self.metric.ComputeMetrics(total_len, batch=self.Settings["batch_size"], std=True)
        Score = results[1] + results[3] + results[5]*100 + results[7]*100
        if not self.training:
            metloc = self.run_dir + "/Model_metrics.txt"
        else:
            metloc = self.modeldir + "/Model_metrics.txt"
        with open(metloc, 'w') as f:
                f.write("Full image:\n")
                f.write(f"PSNR_real_total:                  {results[0]:.2f}  [+/- {results[8]:.2f}]   dB \n")
                f.write(f"PSNR_Generated_total:             {results[1]:.2f}  [+/- {results[9]:.2f}]   dB \n")
                f.write("Defect patch:\n")
                f.write(f"PSNR_real_defect_patch:           {results[2]:.2f}  [+/- {results[10]:.2f}]   dB \n")
                f.write(f"PSNR_Generated_defect_patch:      {results[3]:.2f}  [+/- {results[11]:.2f}]   dB \n")
                f.write("\n")
                f.write("Full image:\n")
                f.write(f"SSIM_real_total:                  {results[4]*100:.2f}  [+/- {results[12]:.2f}]   % \n")
                f.write(f"SSIM_Generated_total:             {results[5]*100:.2f}  [+/- {results[13]:.2f}]   % \n")
                f.write("Defect patch:\n")
                f.write(f"SSIM_real_defect_patch:           {results[6]*100:.2f}  [+/- {results[14]:.2f}]   % \n")
                f.write(f"SSIM_Generated_defect_patch:      {results[7]*100:.2f}  [+/- {results[15]:.2f}]   % \n")
                f.write("\n")
                f.write(f"Model Score:                      {Score:.2f} [+/- 1.42] \n")

        print("Metrics added to Model_metrics.txt file")

if __name__ == '__main__':
    pass