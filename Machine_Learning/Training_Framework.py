import paramiko
import os
from datetime import datetime
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from Losses.Losses import LossFunctions
import time
import sys

#---------------- Helper functions ----------------------
def PIL_concatenate_h(im1, im2, im3):
    out = Image.new('RGB', (im1.width + im2.width + im3.width, im1.height))
    out.paste(im1, (0,0))
    out.paste(im2, (im1.width, 0))    
    out.paste(im3, (im1.width + im2.width, 0))
    return out


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
    

class Training_Framework():
    """
    Framework for training the different networks, should not inherit or mess with pytorch itself, but instead passes the model by assignment to make training more like 
    Legos and less like a novel.
    """

    def __init__(self, Settings, Generator, G_opt, D_opt, Discriminator):
        torch.manual_seed(Settings["seed"])
        np.random.seed(Settings["seed"])
        self.Settings = Settings
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
        
        
        #Initialize loss functions
        losses = LossFunctions(self.device, Discriminator, Settings)

        if Settings["Loss"] == "Hinge_loss":
            self.Discriminator_loss     = losses.Hinge_loss_Discriminator
            self.Generator_loss         = losses.Hinge_loss_Generator
            self.Generator_pixelloss    = losses.Generator_Pixelloss
        if Settings["Loss"] == "WGAN":
            self.Discriminator_loss     = losses.WGAN_Discriminator
            self.Generator_loss         = losses.WGAN_Generator
            self.Generator_pixelloss    = losses.Generator_Pixelloss
        if Settings["Loss"] == "CGAN":
            self.Discriminator_loss     = losses.CGAN_Dual_Encoder_Discriminator
            self.Generator_loss         = losses.CGAN_Generator
            self.Generator_pixelloss    = losses.Generator_Pixelloss
        if Settings["Loss"] == "WGANGP":
            self.Discriminator_loss     = losses.WGANGP_Discriminator
            self.Generator_loss         = losses.WGAN_Generator
            self.Generator_pixelloss    = losses.Generator_Pixelloss

        self.Generator_Deep_Feature_Loss = losses.DeepFeatureLoss



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

        self.Analytics_training("setup", 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.Analytics_validation("setup", 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        self.train_start = time.time()

    def Save_Model(self, epoch):
            #Assuming local and global improvements work in tandem.
            if (epoch == 0) or ((self.Generator_pixel_loss_validation[epoch] + self.Generator_local_pixel_loss_validation[epoch]) < (np.amin(self.Generator_pixel_loss_validation[:epoch]) + np.amin(self.Generator_local_pixel_loss_validation[:epoch]))):
                #Thinking we'll just print for now.
                if not epoch == 0:
                    if (self.Generator_pixel_loss_validation[epoch] < np.amin(self.Generator_pixel_loss_validation[:epoch])):
                        print("model saved on epoch:", epoch, "Due to best pixelloss:", self.Generator_pixel_loss_validation[epoch])
                torch.save(self.Generator.state_dict(), str( self.Modeldir + "/model.pt"))
                torch.save(self.Discriminator.state_dict(), str( self.Modeldir + "/dis_model.pt"))

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


    def Generator_AutoEncoder_updater(self, val=False):  
        self.Generator.zero_grad()    
        # Generator GAN loss
        fake_BB = torch.cat((self.fake_BB, self.real_B), 1)     
        predicted_fake_BB = self.Discriminator(fake_BB)


        loss_GAN_BB = self.Generator_loss(predicted_fake_BB)
        
        #Pixelwise loss
        loss_pixel =  torch.nn.L1Loss(self.fake_BB, self.real_B) #self.Generator_pixelloss(self.fake_BB, self.real_B, self.mask)

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

        return loss_GAN_BB.detach(), loss_pixel.detach(), local_pixelloss.detach(), LatentLoss.detach()
    
    def Generator_Inpainting_updater(self, val=False): 
        self.Generator.zero_grad()       
        # Generator GAN loss
        fake_BA = torch.cat((self.fake_BA, self.real_B), 1)     
        predicted_fake_BA = self.Discriminator(fake_BA)

        loss_GAN_BA = self.Generator_loss(predicted_fake_BA)

        #Pixelwise loss
        _, local_pixelloss =  self.Generator_pixelloss(self.fake_BA, self.real_B, self.mask)

        loss_pixel = torch.zeros(1)
        LatentLoss = torch.zeros(1)
        #Latent Feature loss # only viable for training with dual encoder scheme's 
        #LatentLoss = self.Generator_Deep_Feature_Loss(self.Latent_BA, self.Latent_BB)    

        #Total loss
        Generator_Inpainting_loss = loss_GAN_BA + local_pixelloss * self.Settings["L1__local_loss_weight"]
        
        Total_loss_Generator = Generator_Inpainting_loss
        if not val:
            Total_loss_Generator.backward()
            self.Generator_optimizer.step()

        return loss_GAN_BA.detach(), loss_pixel.detach(), local_pixelloss.detach(), LatentLoss.detach()
     

    def Discriminator_Autoencoder_updater(self, val=False):
        self.Discriminator.zero_grad()
        #Get Critique scores
        fake_BB = torch.cat((self.fake_BB, self.real_B), 1)
        real_AB = torch.cat((self.real_A, self.real_B), 1)     
        pred_real_AB = self.Discriminator(real_AB)
        pred_fake_BB = self.Discriminator(fake_BB.detach())

        #Calculate loss WGAN: loss_real = - torch.mean(real_pred) loss_fake = torch.mean(fake_pred)
        Discriminator_loss = self.Discriminator_loss(pred_fake_BB, pred_real_AB)
        autoencoder_loss = torch.zeros(1)


        if not val:
            Discriminator_loss.backward(retain_graph=True)
            self.Discriminator_optimizer.step()

        return Discriminator_loss.detach(), autoencoder_loss.detach(), pred_real_AB.detach(), pred_fake_BB.detach()

    def Discriminator_Inpainting_updater(self, val=False):
        self.Discriminator.zero_grad()

        #Get Critique scores
        fake_BA = torch.cat((self.fake_BA, self.real_B), 1)   
        real_AB = torch.cat((self.real_A, self.real_B), 1)     
        pred_fake_BA = self.Discriminator(fake_BA.detach())
        pred_real_AB = self.Discriminator(real_AB)

        #Calculate loss # loss_real = - torch.mean(real_pred) loss_fake = torch.mean(fake_pred)
        Total_Discriminator_loss = self.Discriminator_loss(pred_fake_BA, pred_real_AB)
        autoencoder_loss = torch.zeros(1)
        if not val:
            Total_Discriminator_loss.backward(retain_graph=True)
            self.Discriminator_optimizer.step()

        return Total_Discriminator_loss.detach(), autoencoder_loss.detach(), pred_real_AB.detach(), pred_fake_BA.detach()

    
    def FromTorchTraining(self, image):
        #Returns a trainable tensor back into a visual image.
        if self.Settings["Do norm"]:
            return self.Reverse_Normalization(image).permute(1,2,0).mul_(255).clamp(0,255).to("cpu", torch.uint8).numpy()
        else:
            return image.permute(1,2,0).mul_(255).clamp(0,255).to("cpu", torch.uint8).numpy()



    def Generate_validation_images(self, epoch):
        self.Generator.eval()

        if self.real_A.size(0) > 1:
            real_im = self.real_A[0,:,:,:]
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
            PIL_concatenate_h(ma, co, im).save(self.modeltraining_output + "/" + "Image_" + str(epoch) + ".jpg", "JPEG")

        self.Generator.train()

    def validation_run(self, val_loader, epoch):
        with torch.no_grad():

            current_GEN_loss =              torch.zeros(len(val_loader), dtype=torch.float32, device=self.device)
            DeepFeatureloss_arr =           torch.zeros(len(val_loader), dtype=torch.float32, device=self.device)
            current_DIS_loss =              torch.zeros(len(val_loader), dtype=torch.float32, device=self.device)
            pixelloss =                     torch.zeros(len(val_loader), dtype=torch.float32, device=self.device)
            local_pixelloss =               torch.zeros(len(val_loader), dtype=torch.float32, device=self.device)
            Discrim_acc_real_raw =          torch.zeros(len(val_loader), dtype=torch.float32, device=self.device)
            Discrim_acc_fake_raw =          torch.zeros(len(val_loader), dtype=torch.float32, device=self.device)            
            Discrim_auto_loss =             torch.zeros(len(val_loader), dtype=torch.float32, device=self.device)
            Generator_auto_loss =           torch.zeros(len(val_loader), dtype=torch.float32, device=self.device)

            if self.Settings["batch_size"] == 1:
                val_unit = "image(s)"
            else:
                val_unit = "batch(s)"

            #Turn off propagation
            self.Generator.eval()
            self.Generator.requires_grad=False
            self.Discriminator.eval()
            self.Discriminator.requires_grad=False
            
            with tqdm(val_loader, unit=val_unit, leave=False) as tepoch:
                for num, data in enumerate(tepoch):
                    images, defect_images, mask = data
                    if epoch == 0:
                        tepoch.set_description(f"Validation run on Epoch {epoch}/{self.Settings['epochs']}")
                    elif epoch > 0:
                        tepoch.set_description(f"Validation Gen_loss {self.Generator_loss_validation[epoch-1]:.4f} Disc_loss {self.Discriminator_loss_validation[epoch-1]:.4f}")

                    self.real_A = defect_images.to(self.device) #Defect
                    self.real_B = images.to(self.device) #Target
                    self.fake_BA, self.Latent_BA = self.Generator(self.real_A)
                    self.fake_BB, self.Latent_BB = self.Generator(self.real_B)
                    self.mask = mask.to(self.device) # local loss coordinates
                    DIS_loss, DIS_AutoEncoder_loss, predicted_real, predicted_fake = self.Discriminator_updater(val=True)
                    GEN_loss, GEN_AutoEncoder_loss, loss_pixel, loss_pixel_local, DeepFeatureLoss = self.Generator_updater(val=True)

                    #Analytics
                    current_GEN_loss[num] =  GEN_loss
                    current_DIS_loss[num] =  DIS_loss
                    pixelloss[num] =  loss_pixel
                    local_pixelloss[num] =  loss_pixel_local
                    DeepFeatureloss_arr[num] = DeepFeatureLoss
                    Discrim_auto_loss[num] = DIS_AutoEncoder_loss
                    Generator_auto_loss[num] = GEN_AutoEncoder_loss

                    #Self.patch size is torch.size([3])
                    #self.predicted_real size is: torch.size([16, 1, 16, 16])                
                    Discrim_acc_real_raw[num] = torch.mean(predicted_real)
                    Discrim_acc_fake_raw[num] = torch.mean(predicted_fake)

            #Turn on propagation
            self.Generator.train()
            self.Generator.requires_grad=True
            self.Discriminator.train()
            self.Discriminator.requires_grad=True

            #Snapping image from generator during validation
            #if (epoch % 10) == 0:
            #We're now snapping images every epoch 
            self.Generate_validation_images(epoch)

            self.Analytics_validation(epoch, current_GEN_loss, current_DIS_loss, Discrim_acc_real_raw, Discrim_acc_fake_raw, pixelloss, local_pixelloss, DeepFeatureloss_arr, Discrim_auto_loss, Generator_auto_loss)

    def Trainer(self, train_loader, val_loader):
            epochs = tqdm(range(self.Settings["epochs"]), unit="epoch")
            for epoch in epochs:
                epochs.set_description(f"Training the model on epoch {epoch}")
                
                current_GEN_loss =              torch.zeros(len(train_loader), dtype=torch.float32, device=self.device)
                DeepFeatureloss_arr =           torch.zeros(len(train_loader), dtype=torch.float32, device=self.device)
                current_DIS_loss =              torch.zeros(len(train_loader), dtype=torch.float32, device=self.device)
                pixelloss =                     torch.zeros(len(train_loader), dtype=torch.float32, device=self.device)
                local_pixelloss =               torch.zeros(len(train_loader), dtype=torch.float32, device=self.device)
                Discrim_acc_real_raw =          torch.zeros(len(train_loader), dtype=torch.float32, device=self.device)
                Discrim_acc_fake_raw =          torch.zeros(len(train_loader), dtype=torch.float32, device=self.device)
                Discrim_auto_loss =             torch.zeros(len(train_loader), dtype=torch.float32, device=self.device)
                Generator_auto_loss =           torch.zeros(len(train_loader), dtype=torch.float32, device=self.device)

                if self.Settings["batch_size"] == 1:
                    tepoch = tqdm(train_loader, unit='image(s)', leave=False)
                else:
                    tepoch = tqdm(train_loader, unit='batch(s)', leave=False)

                for num, data in enumerate(tepoch):
                    images, defect_images, mask = data
                    if epoch == 0:
                        tepoch.set_description(f"Training on Epoch {epoch}/{self.Settings['epochs']}")
                    elif epoch > 0:
                        tepoch.set_description(f"Training Gen_loss {self.Generator_loss_train[epoch-1]:.4f} Disc_loss {self.Discriminator_loss_train[epoch-1]:.4f}")
                        
                    
                    self.real_A = defect_images.to(self.device) #Defect
                    self.real_B = images.to(self.device) #Target
                    self.fake_BA, self.Latent_BA = self.Generator(self.real_A)
                    self.fake_BB, self.Latent_BB = self.Generator(self.real_B)
                    self.mask = mask.to(self.device) # local loss coordinates
                    # Discriminator return Discriminator_loss.detach(), autoencoder_loss.detach(), pred_real_AB.detach(), pred_fake_BA.detach()
                    #Generator    return loss_GAN_BA.detach(), loss_GAN_BB.detach(), loss_pixel.detach(), local_pixelloss.detach(), LatentLoss.detach()
                    DIS_loss, DIS_AutoEncoder_loss, predicted_real, predicted_fake = self.Discriminator_updater()
                    if num % self.n_crit == 0:
                        GEN_loss, GEN_AutoEncoder_loss, loss_pixel, local_loss_pixel, DeepFeatureloss = self.Generator_updater()

                    #Analytics
                    #This is all assuming batch-size stays at 1
                    current_GEN_loss[num] =  GEN_loss
                    current_DIS_loss[num] =  DIS_loss
                    pixelloss[num] =  loss_pixel
                    local_pixelloss[num] =  local_loss_pixel
                    DeepFeatureloss_arr[num] = DeepFeatureloss
                    Discrim_auto_loss[num] = DIS_AutoEncoder_loss
                    Generator_auto_loss[num] = GEN_AutoEncoder_loss
                    #Self.patch size is torch.size([3])
                    #self.predicted_real size is: torch.size([16, 1, 16, 16])
                    Discrim_acc_real_raw[num] = torch.mean(predicted_real)
                    Discrim_acc_fake_raw[num] = torch.mean(predicted_fake)

                
                #Save per epoch
                self.Analytics_training(epoch, current_GEN_loss, current_DIS_loss, Discrim_acc_real_raw, Discrim_acc_fake_raw, pixelloss, local_pixelloss, DeepFeatureloss_arr, Discrim_auto_loss, Generator_auto_loss)
                self.validation_run(val_loader, epoch)
                self.Save_Model(epoch)
            #Save Analytics to file and create images from analytics    
            self.Save_Analytics()
            self.Create_graphs()
            self.SaveState()
            #Something fishy about metric calc at the end of training. Using the inference.py script until further notice
            #Metrics = Model_Inference(self.Generator, val_loader, self.Settings, self.Modeldir, training=True)
            #Metrics.CreateMetrics()
            #self.CreateMetrics(metric_loader) # Hook this up to the metric loader from the inference class instead
            if self.transmit: # Send finished model to server storage
                print("Sending files")
                self.transmitter.send(self.Modeldir)
                self.transmitter.close()

    def Analytics_training(self, epoch, current_GEN_loss, current_DIS_loss, Discrim_acc_real_raw, Discrim_acc_fake_raw, pixelloss, local_pixelloss, DeepFeatureloss_arr, Discrim_auto_loss, Generator_auto_loss):
        """
        current epoch needs to be the first argument, except when setting up training.  
        """
        if epoch == "setup":
            self.Generator_loss_train = np.zeros(self.Settings["epochs"])
            self.Discriminator_loss_train = np.zeros(self.Settings["epochs"])
            self.Discriminator_accuracy_real_training_raw = np.zeros(self.Settings["epochs"])
            self.Discriminator_accuracy_fake_training_raw = np.zeros(self.Settings["epochs"])
            self.Discriminator_auto_loss_training = np.zeros(self.Settings["epochs"])
            self.Generator_pixel_loss_training = np.zeros(self.Settings["epochs"])
            self.Generator_local_pixel_loss_training = np.zeros(self.Settings["epochs"])
            self.Generator_DeepFeatureLoss_training = np.zeros(self.Settings["epochs"])    
            self.Generator_auto_loss_training = np.zeros(self.Settings["epochs"])

        else:

            #Save per epoch
            """
            current_GEN_loss_in = current_GEN_loss.mean().item()
            current_DIS_loss_in = current_DIS_loss.mean().item()
            Discrim_acc_real_in= Discrim_acc_real_raw.mean().item()
            Discrim_acc_fake_in = Discrim_acc_fake_raw.mean().item()
            pixelloss_in = pixelloss.mean().item()
            """
            self.Generator_loss_train[epoch] = current_GEN_loss.mean().item()
            self.Discriminator_loss_train[epoch] = current_DIS_loss.mean().item()
            self.Discriminator_accuracy_real_training_raw[epoch] = Discrim_acc_real_raw.mean().item()
            self.Discriminator_accuracy_fake_training_raw[epoch] = Discrim_acc_fake_raw.mean().item()
            self.Discriminator_auto_loss_training[epoch] = Discrim_auto_loss.mean().item()
            self.Generator_pixel_loss_training[epoch] = pixelloss.mean().item()
            self.Generator_local_pixel_loss_training[epoch] = local_pixelloss.mean().item()
            self.Generator_DeepFeatureLoss_training[epoch] = DeepFeatureloss_arr.mean().item()
            self.Generator_auto_loss_training[epoch] = Generator_auto_loss.mean().item()

    def Analytics_validation(self, epoch, current_GEN_loss, current_DIS_loss, Discrim_acc_real_raw, Discrim_acc_fake_raw, pixelloss, local_pixelloss, DeepFeatureloss_arr, Discrim_auto_loss, Generator_auto_loss):
        """
        current epoch needs to be the first argument, except when setting up training. 
        """
        if epoch == "setup":
            self.Generator_loss_validation = np.zeros(self.Settings["epochs"])
            self.Discriminator_loss_validation = np.zeros(self.Settings["epochs"])
            self.Discriminator_accuracy_real_validation_raw = np.zeros(self.Settings["epochs"])
            self.Discriminator_accuracy_fake_validation_raw = np.zeros(self.Settings["epochs"])    
            self.Discriminator_auto_loss_validation = np.zeros(self.Settings["epochs"])
            self.Generator_pixel_loss_validation = np.zeros(self.Settings["epochs"])    
            self.Generator_local_pixel_loss_validation = np.zeros(self.Settings["epochs"])    
            self.Generator_DeepFeatureLoss_validation = np.zeros(self.Settings["epochs"])    
            self.Generator_auto_loss_validation = np.zeros(self.Settings["epochs"])
        else:
            """
            current_GEN_loss_in = current_GEN_loss.mean().item()
            current_DIS_loss_in = current_DIS_loss.mean().item()
            Discrim_acc_real_in = Discrim_acc_real_raw.mean().item()
            Discrim_acc_fake_in = Discrim_acc_fake_raw.mean().item()
            pixelloss_in = pixelloss.mean().item()
            """
            self.Generator_loss_validation[epoch] = current_GEN_loss.mean().item()
            self.Discriminator_loss_validation[epoch] = current_DIS_loss.mean().item()
            self.Discriminator_accuracy_real_validation_raw[epoch] = Discrim_acc_real_raw.mean().item()
            self.Discriminator_accuracy_fake_validation_raw[epoch] = Discrim_acc_fake_raw.mean().item()
            self.Discriminator_auto_loss_validation[epoch] = Discrim_auto_loss.mean().item()
            self.Generator_pixel_loss_validation[epoch] = pixelloss.mean().item()
            self.Generator_local_pixel_loss_validation[epoch] = local_pixelloss.mean().item()
            self.Generator_DeepFeatureLoss_validation[epoch] = DeepFeatureloss_arr.mean().item()
            self.Generator_auto_loss_validation[epoch] = Generator_auto_loss.mean().item()

    def Save_Analytics(self):
        np.savez(self.Modeldir + '/Analytics.npz', (self.Generator_loss_validation,
                                self.Discriminator_loss_validation,
                                self.Discriminator_accuracy_real_validation_raw,
                                self.Discriminator_accuracy_fake_validation_raw,
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
                                ))




    def Create_graphs(self):
        xaxis = np.arange(1, self.Generator_loss_validation.shape[0]+1)
        plt.plot(xaxis, self.Generator_loss_train, label="Generator loss training")
        plt.plot(xaxis, self.Generator_loss_validation, label="Generator loss validation")
        plt.xlabel("epochs")
        plt.ylabel("Loss ")
        plt.title("Generator loss curves")
        plt.legend()
        plt.savefig(self.Modeldir + "/Generator_loss_curves.png")
        plt.clf() # clear the plot

        plt.plot(xaxis, self.Discriminator_loss_train, label="Discriminator loss training")    
        plt.plot(xaxis, self.Discriminator_loss_validation, label="Discriminator loss validation")
        plt.xlabel("epochs")
        plt.ylabel("Loss ")
        plt.title("Discriminator loss curves")
        plt.legend()
        plt.savefig(self.Modeldir + "/Discriminator_loss_curves.png")
        plt.clf() # clear the plot

        plt.plot(xaxis, self.Generator_loss_train, label="Generator loss training")
        plt.plot(xaxis, self.Generator_loss_validation, label="Generator loss validation")
        plt.plot(xaxis, self.Discriminator_loss_train, label="Discriminator loss training")    
        plt.plot(xaxis, self.Discriminator_loss_validation, label="Discriminator loss validation")
        plt.xlabel("epochs")
        plt.ylabel("Loss ")
        plt.title("Combined loss curves")
        plt.legend()
        plt.savefig(self.Modeldir + "/Combined_loss_curves.png")
        plt.clf() # clear the plot

        plt.plot(xaxis, self.Generator_DeepFeatureLoss_training, label="Training")
        plt.plot(xaxis, self.Generator_DeepFeatureLoss_validation, label="Validation")
        plt.xlabel("epochs")
        plt.ylabel("Loss ")
        plt.title("Deep Feature Loss ")
        plt.legend()
        plt.savefig(self.Modeldir + "/DeepFeatureLoss.png")
        plt.clf() # clear the plot

        plt.plot(xaxis, self.Discriminator_auto_loss_training, label="Training")
        plt.plot(xaxis, self.Discriminator_auto_loss_validation, label="Validation")
        plt.xlabel("epochs")
        plt.ylabel("Loss ")
        plt.title("Discriminator Autoencoder Loss ")
        plt.legend()
        plt.savefig(self.Modeldir + "/Discriminator_AutoEncoder_loss.png")
        plt.clf() # clear the plot

        plt.plot(xaxis, self.Generator_auto_loss_training, label="Training")
        plt.plot(xaxis, self.Generator_auto_loss_validation, label="Validation")
        plt.xlabel("epochs")
        plt.ylabel("Loss ")
        plt.title("Generator Autoencoder Loss ")
        plt.legend()
        plt.savefig(self.Modeldir + "/Generator_AutoEncoder_loss.png")
        plt.clf() # clear the plot

        #Implement this in analytics
        plt.plot(xaxis, self.Discriminator_accuracy_real_training_raw, label="Discriminator accuracy real training")
        plt.plot(xaxis, self.Discriminator_accuracy_fake_training_raw, label="Discriminator accuracy fake training")
        plt.plot(xaxis, self.Discriminator_accuracy_real_validation_raw, label="Discriminator accuracy real validation")
        plt.plot(xaxis, self.Discriminator_accuracy_fake_validation_raw, label="Discriminator accuracy fake validation")
        plt.xlabel("epochs")
        plt.ylabel("no unit")
        plt.title("Discriminator accuracy raw data")
        plt.legend()
        plt.savefig(self.Modeldir + "/discriminator_accuracy_raw_curves.png")
        plt.clf()
        #Implement this in analytics
        plt.plot(xaxis, self.Generator_pixel_loss_training, label="Training")
        plt.plot(xaxis, self.Generator_pixel_loss_validation, label="Validation")
        plt.xlabel("epochs")
        plt.ylabel("L1-loss")
        plt.title("Generator combined pixel loss")
        plt.legend()
        plt.savefig(self.Modeldir + "/generator_pixel_loss.png")
        plt.clf()
        #Implement this in analytics
        plt.plot(xaxis, self.Generator_local_pixel_loss_training, label="Training")
        plt.plot(xaxis, self.Generator_local_pixel_loss_validation, label="Validation")
        plt.xlabel("epochs")
        plt.ylabel("L1-loss")
        plt.title("Generator combined local pixel loss")
        plt.legend()
        plt.savefig(self.Modeldir + "/generator_local_pixel_loss.png")
        plt.clf()

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
        
    def Inference_run(self, runs=8):
        """
        Does an inference run on the Model for three images
        """
        self.model.eval()
        loader = tqdm(range(runs))
        with torch.no_grad():
            for run in loader:
                loader.set_description(f"Running {run+1}/{runs} images completed")
                _ , defect_image, mask = next(iter(self.dataloader))
                real_A = defect_image.to(self.device)
                fake_B, _ = self.model(real_A.clone())
                im = Image.fromarray(self.FromTorchTraining(fake_B.squeeze(0)))
                co = Image.fromarray(self.FromTorchTraining(real_A.squeeze(0)))
                ma = Image.fromarray(self.FromTorchTraining(mask.squeeze(0).int()))
                PIL_concatenate_h(ma, co, im).save(self.run_dir + "/output/image_" + str(run) + ".jpg")

        print("Done!")
        print("All results saved to:")
        print(self.run_dir)

    def CenteringAlgorithm(self, BoxSize, BoundingBox, Y, X):
        """
        Returns a larger bounding box centered on the defect block
        """

        x = torch.round(X + 0.5*BoxSize - 0.5*BoundingBox).to(torch.uint8).clamp(0, 256 - BoundingBox)
        y = torch.round(Y + 0.5*BoxSize - 0.5*BoundingBox).to(torch.uint8).clamp(0, 256 - BoundingBox)

        return y,x

    def CreateMetrics(self):
        total_len = 500 # manually selected to not take too much time.
        with torch.no_grad():
            self.model.eval()
            total_images = total_len
            PSNR_real_values = np.zeros((total_images, 3))
            PSNR_fake_values = np.zeros((total_images, 3))
            SSIM_real_values = np.zeros((total_images, 3))
            SSIM_fake_values = np.zeros((total_images, 3))

            PSNR_real_values_p = np.zeros((total_images, 3))
            PSNR_fake_values_p = np.zeros((total_images, 3))
            SSIM_real_values_p = np.zeros((total_images, 3))
            SSIM_fake_values_p = np.zeros((total_images, 3))
            lbar = tqdm(range(total_images))
            for num in lbar:
                images, defect_images, defect_mask = next(iter(self.dataloader))
                lbar.set_description(f"Running metrics {num+1}/{total_images} images")

                if num > (total_len - 1):
                    break
                
                #Load images and run inference 
                real_A = defect_images.to(self.device) #Defect
                real_B = images.to(self.device) #Target 
                fake_B, _ = self.model(real_A.clone())
                mask = defect_mask.to(self.device)

                #Assign defect regions and convert
                local_fake_B = self.FromTorchTraining(torch.masked_select(fake_B, ~mask).view(3,8,8).squeeze(0))# defect-region
                local_real_B = self.FromTorchTraining(torch.masked_select(real_B, ~mask).view(3,8,8).squeeze(0))# defect-region
                local_real_A = self.FromTorchTraining(torch.masked_select(real_A, ~mask).view(3,8,8).squeeze(0))# defect-region

                fake_B = self.FromTorchTraining(fake_B.squeeze(0))
                real_B = self.FromTorchTraining(real_B.squeeze(0))
                real_A = self.FromTorchTraining(real_A.squeeze(0))

                for channel in range(3):
                    PSNR_real_values[num, channel]       = PSNR(real_B[:,:,channel], real_A[:,:,channel], data_range=255)
                    PSNR_fake_values[num, channel]       = PSNR(real_B[:,:,channel], fake_B[:,:,channel], data_range=255)
                    PSNR_real_values_p[num, channel]     = PSNR(local_real_B[:,:,channel], local_real_A[:,:,channel], data_range=255)
                    PSNR_fake_values_p[num, channel]     = PSNR(local_real_B[:,:,channel], local_fake_B[:,:,channel], data_range=255)


                    SSIM_real_values_p[num, channel]     =  SSIM(local_real_B[:,:,channel], local_real_A[:,:,channel], data_range=255)
                    SSIM_fake_values_p[num, channel]     =  SSIM(local_real_B[:,:,channel], local_fake_B[:,:,channel], data_range=255)
                    SSIM_real_values[num, channel]       =  SSIM(real_B[:,:,channel], real_A[:,:,channel], data_range=255)
                    SSIM_fake_values[num, channel]       =  SSIM(real_B[:,:,channel], fake_B[:,:,channel], data_range=255)

            
            PSNR_fake_mean = np.ma.masked_invalid(PSNR_fake_values).mean(axis=(1,0))
            PSNR_real_mean = np.ma.masked_invalid(PSNR_real_values).mean(axis=(1,0))
            SSIM_fake_mean = np.ma.masked_invalid(SSIM_fake_values).mean(axis=(1,0))
            SSIM_real_mean = np.ma.masked_invalid(SSIM_real_values).mean(axis=(1,0))
            PSNR_fake_mean_p = np.ma.masked_invalid(PSNR_fake_values_p).mean(axis=(1,0))
            PSNR_real_mean_p = np.ma.masked_invalid(PSNR_real_values_p).mean(axis=(1,0))
            SSIM_fake_mean_p = np.ma.masked_invalid(SSIM_fake_values_p).mean(axis=(1,0))
            SSIM_real_mean_p = np.ma.masked_invalid(SSIM_real_values_p).mean(axis=(1,0))

            if not self.training:
                metloc = self.run_dir + "/Model_metrics.txt"
            else:
                metloc = self.modeldir + "/Model_metrics.txt"
            with open(metloc, 'w') as f:
                    f.write("Full image:\n")
                    f.write(f"PSNR_real_total:                  {PSNR_real_mean:.2f}    dB \n")
                    f.write(f"PSNR_Generated_total:             {PSNR_fake_mean:.2f}    dB \n")
                    f.write("Defect patch:\n")
                    f.write(f"PSNR_real_defect_patch:           {PSNR_real_mean_p:.2f}    dB \n")
                    f.write(f"PSNR_Generated_defect_patch:      {PSNR_fake_mean_p:.2f}    dB \n")
                    f.write("\n")
                    f.write("Full image:\n")
                    f.write(f"SSIM_real_total:                  {SSIM_real_mean*100:.2f}    % \n")
                    f.write(f"SSIM_Generated_total:             {SSIM_fake_mean*100:.2f}    % \n")
                    f.write("Defect patch:\n")
                    f.write(f"SSIM_real_defect_patch:           {SSIM_real_mean_p*100:.2f}    % \n")
                    f.write(f"SSIM_Generated_defect_patch:      {SSIM_fake_mean_p*100:.2f}    % \n")
            print("Metrics added to Model_metrics.txt file")

if __name__ == '__main__':
    pass