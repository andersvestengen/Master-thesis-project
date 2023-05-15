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
import time
import sys

#---------------- Helper functions ----------------------
def PIL_concatenate_h(im1, im2):
    out = Image.new('RGB', (im1.width + im2.width, im1.height))
    out.paste(im1, (0,0))
    out.paste(im2, (im1.width, 0))
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

    def __init__(self, Settings, Generator, G_opt, D_opt, GAN_loss, pixelwise_loss, Discriminator):
        torch.manual_seed(Settings["seed"])
        np.random.seed(Settings["seed"])
        self.Settings = Settings
        self.Generator = Generator
        self.Generator_optimizer = G_opt
        self.GAN_loss = GAN_loss
        self.pixelwise_loss = pixelwise_loss
        self.Discriminator = Discriminator
        self.image_transform = transforms.ToPILImage()
        self.Discriminator_optimizer = D_opt
        self.device = self.Settings["device"]
        self.Generator_loss = 0
        self.Discriminator_loss = 0
        self.Reverse_Normalization = NormalizeInverse(self.Settings["Data_mean"], self.Settings["Data_std"])
        self.patch = torch.tensor((1, self.Settings["ImageHW"] // 2 ** 4, self.Settings["ImageHW"] // 2 ** 4), device=self.device)
        self.transmit = False # No reason to do file transfer in the future

        """
         decision = input("would you like to send model to server? [y/n]: ")

        if decision == "y":
            self.transmit = True
            self.transmitter = FileSender()       
        """
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

        self.Analytics_training("setup", 0, 0, 0, 0, 0)
        self.Analytics_validation("setup", 0, 0, 0, 0, 0)
        
        self.train_start = time.time()

    def Save_Model(self, epoch):
            if (epoch == 0) or (self.Generator_loss_validation[epoch] < np.min(self.Generator_loss_validation[:epoch])) or (self.Generator_pixel_loss_validation[epoch] < np.min(self.Generator_pixel_loss_validation[:epoch])):
                torch.save(self.Generator.state_dict(), str( self.Modeldir + "/model.pt"))

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

    def CenteringAlgorithm(self, BoxSize, BoundingBox, Y, X):
        """
        Returns a larger bounding box centered on the defect block
        """

        x = torch.round(X + 0.5*BoxSize - 0.5*BoundingBox).to(self.device, torch.uint8).clamp(0, 256 - BoundingBox)
        y = torch.round(Y + 0.5*BoxSize - 0.5*BoundingBox).to(self.device, torch.uint8).clamp(0, 256 - BoundingBox)

        return y,x


    def Make_Label_Tensor(self, tensor_size, bool_val):
        """
        Return a label tensor with size tensor_size and values bool_val
        """       

        if bool_val:
            label_tensor = torch.tensor(1, device=self.device, dtype=torch.float32, requires_grad=False)
        else:
            label_tensor = torch.tensor(0, device=self.device, dtype=torch.float32)

        return label_tensor.expand_as(tensor_size)


    def Generator_updater(self, val=False): 
        self.Generator.zero_grad()
        #valid = torch.ones((self.Settings["batch_size"], *self.patch), requires_grad=False).to(self.device)
        
        # Generator loss
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)     
        predicted_fake = self.Discriminator(fake_AB)
        valid = self.Make_Label_Tensor(predicted_fake, True)
        loss_GAN = self.GAN_loss(predicted_fake, valid)
        
        #Pixelwise loss
        SampleY, SampleX, BoxSize = self.defect_coordinates[0]
        L1_loss_region = (BoxSize * int(self.Settings["Loss_region_Box_mult"])).to(self.device) # trying standard 30x30 loss box
        SampleY, SampleX = self.CenteringAlgorithm(BoxSize, L1_loss_region, SampleY, SampleX)
        loss_pixel = self.pixelwise_loss(self.fake_B, self.real_B)
        local_pixelloss = self.pixelwise_loss(self.fake_B[:,:,SampleY:SampleY+L1_loss_region,SampleX:SampleX+L1_loss_region], self.real_B[:,:,SampleY:SampleY+L1_loss_region,SampleX:SampleX+L1_loss_region])
        
        #Total loss
        Total_loss_Generator = loss_GAN + self.Settings["L1_loss_weight"] * loss_pixel + self.Settings["L1__local_loss_weight"] * local_pixelloss
        
        if not val:
            Total_loss_Generator.backward()
            self.Generator_optimizer.step()

        return loss_GAN.detach(), (loss_pixel + local_pixelloss).detach()

    def Discriminator_updater(self, val=False):
        #valid = torch.ones((self.Settings["batch_size"], *self.patch), requires_grad=False).to(self.device)
        #fake = torch.zeros((self.Settings["batch_size"], *self.patch), requires_grad=False).to(self.device)
        
        #Real loss
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        predicted_real = self.Discriminator(real_AB)
        
        valid = self.Make_Label_Tensor(predicted_real, True)
        fake = self.Make_Label_Tensor(predicted_real, False)
        loss_real = self.GAN_loss(predicted_real, valid)

        #Fake loss
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)        
        predicted_fake = self.Discriminator(fake_AB.detach())
        loss_fake = self.GAN_loss(predicted_fake, fake)

        #Total loss and backprop
        Total_loss_Discriminator = 0.5 * (loss_real + loss_fake)
        if not val: 
            Total_loss_Discriminator.backward() # backward run        
            self.Discriminator_optimizer.step() # step

        return Total_loss_Discriminator.detach(), predicted_real.detach(), predicted_fake.detach()

    
    def FromTorchTraining(self, image):
        #Returns a trainable tensor back into a visual image.
        if self.Settings["Do norm"]:
            return self.Reverse_Normalization(image).permute(1,2,0).mul_(255).clamp(0,255).to("cpu", torch.uint8).numpy()
        else:
            return image.permute(1,2,0).mul_(255).clamp(0,255).to("cpu", torch.uint8).numpy()



    def Generate_validation_images(self, epoch):
        self.Generator.eval()

        if self.real_A.size(0) > 1:
            real_im = self.real_A[0,:,:,:].clone()
        else:
            real_im = self.real_A.clone()

        with torch.no_grad():
            if self.real_A.size(0) > 1:
                fake_B = self.Generator(real_im.clone().unsqueeze(0))
                im = Image.fromarray(self.FromTorchTraining(fake_B.squeeze(0)))
                co = Image.fromarray(self.FromTorchTraining(real_im))
            else:
                fake_B = self.Generator(real_im.clone())
                im = Image.fromarray(self.FromTorchTraining(fake_B.squeeze(0)))
                co = Image.fromarray(self.FromTorchTraining(real_im.squeeze(0)))
            PIL_concatenate_h(co, im).save(self.modeltraining_output + "/" + "Image_" + str(epoch) + ".jpg", "JPEG")

        self.Generator.train()

    def validation_run(self, val_loader, epoch):
        with torch.no_grad():

            current_GEN_loss =      torch.zeros(len(val_loader)*self.Settings["batch_size"], dtype=torch.float32, device=self.device)
            current_DIS_loss =      torch.zeros(len(val_loader)*self.Settings["batch_size"], dtype=torch.float32, device=self.device)
            pixelloss =             torch.zeros(len(val_loader)*self.Settings["batch_size"], dtype=torch.float32, device=self.device)
            Discrim_acc_real_raw =  torch.zeros(len(val_loader)*self.Settings["batch_size"], dtype=torch.float32, device=self.device)
            Discrim_acc_fake_raw =  torch.zeros(len(val_loader)*self.Settings["batch_size"], dtype=torch.float32, device=self.device)

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
                    images, defect_images, defect_coordinates = data
                    if epoch == 0:
                        tepoch.set_description(f"Validation run on Epoch {epoch}/{self.Settings['epochs']}")
                    elif epoch > 0:
                        tepoch.set_description(f"Validation Gen_loss {self.Generator_loss_validation[epoch-1]:.4f} Disc_loss {self.Discriminator_loss_validation[epoch-1]:.4f}")

                    self.real_A = defect_images.to(self.device) #Defect
                    self.real_B = images.to(self.device) #Target
                    self.fake_B = self.Generator(self.real_A)
                    self.defect_coordinates = defect_coordinates.to(self.device) # local loss coordinates
                    DIS_loss, predicted_real, predicted_fake = self.Discriminator_updater(val=True)
                    GEN_loss, loss_pixel = self.Generator_updater(val=True)

                    #Analytics
                    current_GEN_loss[num: num + self.Settings["batch_size"]] =  GEN_loss
                    current_DIS_loss[num: num + self.Settings["batch_size"]] =  DIS_loss
                    pixelloss[num: num + self.Settings["batch_size"]] =  loss_pixel

                    #Self.patch size is torch.size([3])
                    #self.predicted_real size is: torch.size([16, 1, 16, 16])                
                    Discrim_acc_real_raw[num: num + self.Settings["batch_size"]] = torch.mean(predicted_real, (2,3)).squeeze(1)
                    Discrim_acc_fake_raw[num: num + self.Settings["batch_size"]] = torch.mean(predicted_fake, (2,3)).squeeze(1)

            #Turn on propagation
            self.Generator.train()
            self.Generator.requires_grad=True
            self.Discriminator.train()
            self.Discriminator.requires_grad=True

            #Snapping image from generator during validation
            if (epoch % 10) == 0:
                self.Generate_validation_images(epoch)

            self.Analytics_validation(epoch, current_GEN_loss, current_DIS_loss, Discrim_acc_real_raw, Discrim_acc_fake_raw, pixelloss)

    def Trainer(self, train_loader, val_loader):
            epochs = tqdm(range(self.Settings["epochs"]), unit="epoch")
            for epoch in epochs:
                epochs.set_description(f"Training the model on epoch {epoch}")
                current_GEN_loss =      torch.zeros(len(val_loader)*self.Settings["batch_size"], dtype=torch.float32, device=self.device)
                current_DIS_loss =      torch.zeros(len(val_loader)*self.Settings["batch_size"], dtype=torch.float32, device=self.device)
                pixelloss =             torch.zeros(len(val_loader)*self.Settings["batch_size"], dtype=torch.float32, device=self.device)
                Discrim_acc_real_raw =  torch.zeros(len(val_loader)*self.Settings["batch_size"], dtype=torch.float32, device=self.device)
                Discrim_acc_fake_raw =  torch.zeros(len(val_loader)*self.Settings["batch_size"], dtype=torch.float32, device=self.device)

                if self.Settings["batch_size"] == 1:
                    tepoch = tqdm(train_loader, unit='image(s)', leave=False)
                else:
                    tepoch = tqdm(train_loader, unit='batch(s)', leave=False)

                for num, data in enumerate(tepoch):
                    images, defect_images, defect_coordinates = data
                    if epoch == 0:
                        tepoch.set_description(f"Training on Epoch {epoch}/{self.Settings['epochs']}")
                    elif epoch > 0:
                        tepoch.set_description(f"Training Gen_loss {self.Generator_loss_train[epoch-1]:.4f} Disc_loss {self.Discriminator_loss_train[epoch-1]:.4f}")
                        

                    self.real_A = defect_images.to(self.device) #Defect
                    self.real_B = images.to(self.device) #Target
                    self.fake_B = self.Generator(self.real_A)
                    self.defect_coordinates = defect_coordinates.to(self.device) # local loss coordinates
                    DIS_loss, predicted_real, predicted_fake = self.Discriminator_updater()
                    GEN_loss, loss_pixel, = self.Generator_updater()

                    #Analytics
                    #This is all assuming batch-size stays at 1
                    current_GEN_loss[num: num + self.Settings["batch_size"]] =  GEN_loss
                    current_DIS_loss[num: num + self.Settings["batch_size"]] =  DIS_loss
                    pixelloss[num: num + self.Settings["batch_size"]] =  loss_pixel

                    #Self.patch size is torch.size([3])
                    #self.predicted_real size is: torch.size([16, 1, 16, 16])
                    Discrim_acc_real_raw[num: num + self.Settings["batch_size"]] = torch.mean(predicted_real, (2,3)).squeeze(1)
                    Discrim_acc_fake_raw[num: num + self.Settings["batch_size"]] = torch.mean(predicted_fake, (2,3)).squeeze(1)

                
                #Save per epoch
                self.Analytics_training(epoch, current_GEN_loss, current_DIS_loss, Discrim_acc_real_raw, Discrim_acc_fake_raw, pixelloss)
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

    def Analytics_training(self, epoch, current_GEN_loss, current_DIS_loss, Discrim_acc_real_raw, Discrim_acc_fake_raw, pixelloss):
        """
        current epoch needs to be the first argument, except when setting up training.  
        """
        if epoch == "setup":
            self.Generator_loss_train = np.zeros(self.Settings["epochs"])
            self.Discriminator_loss_train = np.zeros(self.Settings["epochs"])
            self.Discriminator_accuracy_real_training = np.zeros(self.Settings["epochs"])
            self.Discriminator_accuracy_fake_training = np.zeros(self.Settings["epochs"])
            self.Discriminator_accuracy_real_training_raw = np.zeros(self.Settings["epochs"])
            self.Discriminator_accuracy_fake_training_raw = np.zeros(self.Settings["epochs"])
            self.Generator_pixel_loss_training = np.zeros(self.Settings["epochs"])

        else:

            #Save per epoch
            current_GEN_loss_in = current_GEN_loss.mean().item()
            current_DIS_loss_in = current_DIS_loss.mean().item()
            Discrim_acc_real_in= Discrim_acc_real_raw.mean().item()
            Discrim_acc_fake_in = Discrim_acc_fake_raw.mean().item()
            pixelloss_in = pixelloss.mean().item()

            self.Generator_loss_train[epoch] = current_GEN_loss_in
            self.Discriminator_loss_train[epoch] = current_DIS_loss_in
            self.Discriminator_accuracy_real_training_raw[epoch] = Discrim_acc_real_in
            self.Discriminator_accuracy_fake_training_raw[epoch] = Discrim_acc_fake_in
            self.Generator_pixel_loss_training[epoch] = pixelloss_in

    def Analytics_validation(self, epoch, current_GEN_loss, current_DIS_loss, Discrim_acc_real_raw, Discrim_acc_fake_raw, pixelloss):
        """
        current epoch needs to be the first argument, except when setting up training. 
        """
        if epoch == "setup":
            self.Generator_loss_validation = np.zeros(self.Settings["epochs"])
            self.Discriminator_loss_validation = np.zeros(self.Settings["epochs"])
            self.Discriminator_accuracy_real_validation = np.zeros(self.Settings["epochs"])
            self.Discriminator_accuracy_fake_validation = np.zeros(self.Settings["epochs"])
            self.Discriminator_accuracy_real_validation_raw = np.zeros(self.Settings["epochs"])
            self.Discriminator_accuracy_fake_validation_raw = np.zeros(self.Settings["epochs"])    
            self.Generator_pixel_loss_validation = np.zeros(self.Settings["epochs"])    

        else:
            current_GEN_loss_in = current_GEN_loss.mean().item()
            current_DIS_loss_in = current_DIS_loss.mean().item()
            Discrim_acc_real_in = Discrim_acc_real_raw.mean().item()
            Discrim_acc_fake__in = Discrim_acc_fake_raw.mean().item()
            pixelloss_in = pixelloss.mean().item()

            self.Generator_loss_validation[epoch] = current_GEN_loss_in
            self.Discriminator_loss_validation[epoch] = current_DIS_loss_in
            self.Discriminator_accuracy_real_validation_raw[epoch] = Discrim_acc_real_in
            self.Discriminator_accuracy_fake_validation_raw[epoch] = Discrim_acc_real_in
            self.Generator_pixel_loss_validation[epoch] = pixelloss_in

    def Save_Analytics(self):
        np.savez(self.Modeldir + '/Analytics.npz', (self.Generator_loss_validation,
                                self.Discriminator_loss_validation,
                                self.Discriminator_accuracy_real_validation,
                                self.Discriminator_accuracy_fake_validation,
                                self.Discriminator_accuracy_real_validation_raw,
                                self.Discriminator_accuracy_fake_validation_raw,
                                self.Generator_pixel_loss_validation,
                                self.Generator_loss_train,
                                self.Generator_pixel_loss_training,
                                self.Discriminator_loss_train,
                                self.Discriminator_accuracy_real_training,
                                self.Discriminator_accuracy_fake_training,
                                self.Discriminator_accuracy_real_training_raw,
                                self.Discriminator_accuracy_fake_training_raw
                                ))



    def Create_graphs(self):
        xaxis = np.arange(1, self.Generator_loss_validation.shape[0]+1)
        plt.plot(xaxis, self.Generator_loss_train, label="Generator loss training")
        plt.plot(xaxis, self.Generator_loss_validation, label="Generator loss validation")
        plt.xlabel("epochs")
        plt.ylabel("Loss")
        plt.title("Generator loss curves")
        plt.legend()
        plt.savefig(self.Modeldir + "/Generator_loss_curves.png")
        plt.clf() # clear the plot

        xaxis = np.arange(1, self.Discriminator_loss_validation.shape[0]+1)
        plt.plot(xaxis, self.Discriminator_loss_train, label="Discriminator loss training")    
        plt.plot(xaxis, self.Discriminator_loss_validation, label="Discriminator loss validation")
        plt.xlabel("epochs")
        plt.ylabel("Loss")
        plt.title("Discriminator loss curves")
        plt.legend()
        plt.savefig(self.Modeldir + "/Discriminator_loss_curves.png")
        plt.clf() # clear the plot

        xaxis = np.arange(1, self.Discriminator_loss_validation.shape[0]+1)
        plt.plot(xaxis, self.Generator_loss_train, label="Generator loss training")
        plt.plot(xaxis, self.Generator_loss_validation, label="Generator loss validation")
        plt.plot(xaxis, self.Discriminator_loss_train, label="Discriminator loss training")    
        plt.plot(xaxis, self.Discriminator_loss_validation, label="Discriminator loss validation")
        plt.xlabel("epochs")
        plt.ylabel("Loss")
        plt.title("Combined loss curves")
        plt.legend()
        plt.savefig(self.Modeldir + "/Combined_loss_curves.png")
        plt.clf() # clear the plot

        plt.plot(xaxis, self.Discriminator_accuracy_real_training*100, label="Discriminator accuracy real training")
        plt.plot(xaxis, self.Discriminator_accuracy_fake_training*100, label="Discriminator accuracy fake training")
        plt.plot(xaxis, self.Discriminator_accuracy_real_validation*100, label="Discriminator accuracy real validation")
        plt.plot(xaxis, self.Discriminator_accuracy_fake_validation*100, label="Discriminator accuracy fake validation")
        plt.xlabel("epochs")
        plt.ylabel("Percentage [%]")
        plt.title("Discriminator accuracy")
        plt.legend()
        plt.savefig(self.Modeldir + "/discriminator_accuracy_curves.png")
        plt.clf()

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
        plt.plot(xaxis, self.Generator_pixel_loss_training*100, label="Training")
        plt.plot(xaxis, self.Generator_pixel_loss_validation*100, label="Validation")
        plt.xlabel("epochs")
        plt.ylabel("L1-loss")
        plt.title("Generator combined pixel loss")
        plt.legend()
        plt.savefig(self.Modeldir + "/generator_pixel_loss.png")
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
                _ , defect_image, _ = next(iter(self.dataloader))
                real_A = defect_image.to(self.device)
                fake_B = self.model(real_A.clone())
                im = Image.fromarray(self.FromTorchTraining(fake_B.squeeze(0)))
                co = Image.fromarray(self.FromTorchTraining(real_A.squeeze(0)))
                PIL_concatenate_h(co, im).save(self.run_dir + "/output/image_" + str(run) + ".jpg")

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
                images, defect_images, coordinates = next(iter(self.dataloader))
                lbar.set_description(f"Running metrics {num+1}/{total_images} images")

                if num > (total_len - 1):
                    break
                
                real_A = defect_images.to(self.device) #Defect
                real_B = images.to(self.device) #Target 
                fake_B = self.model(real_A.clone())

                real_A = self.FromTorchTraining(real_A.squeeze(0))
                real_B = self.FromTorchTraining(real_B.squeeze(0))
                fake_B = self.FromTorchTraining(fake_B.squeeze(0))
    
                #Getting patch coordinates
                SampleY, SampleX, BoxSize = coordinates[0]
                #BoxSize = self.BoxSet[1] * int(self.Settings["Loss_region_Box_mult"])
                if BoxSize < 7:
                    L1_loss_region = 7
                else:
                    L1_loss_region = BoxSize
                SampleY, SampleX = self.CenteringAlgorithm(BoxSize, L1_loss_region, SampleY, SampleX)
                for channel in range(3):
                    PSNR_real_values[num, channel]       = PSNR(real_B[:,:,channel], real_A[:,:,channel], data_range=255)
                    PSNR_fake_values[num, channel]       = PSNR(real_B[:,:,channel], fake_B[:,:,channel], data_range=255)
                    PSNR_real_values_p[num, channel]     = PSNR(real_B[SampleY:SampleY+L1_loss_region,SampleX:SampleX+L1_loss_region,channel], real_A[SampleY:SampleY+L1_loss_region,SampleX:SampleX+L1_loss_region,channel], data_range=255)
                    PSNR_fake_values_p[num, channel]     = PSNR(real_B[SampleY:SampleY+L1_loss_region,SampleX:SampleX+L1_loss_region,channel], fake_B[SampleY:SampleY+L1_loss_region,SampleX:SampleX+L1_loss_region,channel], data_range=255)


                    SSIM_real_values_p[num, channel]     =  SSIM(real_B[SampleY:SampleY+L1_loss_region,SampleX:SampleX+L1_loss_region,channel], real_A[SampleY:SampleY+L1_loss_region,SampleX:SampleX+L1_loss_region,channel], data_range=255)
                    SSIM_fake_values_p[num, channel]     =  SSIM(real_B[SampleY:SampleY+L1_loss_region,SampleX:SampleX+L1_loss_region,channel], fake_B[SampleY:SampleY+L1_loss_region,SampleX:SampleX+L1_loss_region,channel], data_range=255)
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