import paramiko
import os
from datetime import datetime
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

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


    def close(self):
        self.ftr.close()
        self.cli.close()
        print("SSH and SFTP closed")




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
        self.patch = (1, self.Settings["ImageHW"] // 2 ** 4, self.Settings["ImageHW"] // 2 ** 4)
        self.transmit = False
        decision = input("would you like to send model to server? [y/n]: ")

        if decision == "y":
            self.transmit = True
            self.transmitter = FileSender()

        # Set the working Directory
        if not self.Settings["dataset_loc"]:
            self.workingdir = os.getcwd()
        else:
            self.workingdir = self.Settings["dataset_loc"]

        #Create the directory of the model (Look back at this for batch training.)
        time = str(datetime.now())
        stamp = time[:-16] + "_" + time[11:-7].replace(":", "-")
        if self.Settings["ModelTrainingName"] is not None:
            self.Modeldir = self.workingdir +  "/Trained_Models/" + self.Settings["ModelTrainingName"] + " (" + stamp +")"
        else:
            self.Modeldir = self.workingdir + "/Trained_Models/" + "GAN_Model" + " " + stamp
            
        os.makedirs(self.Modeldir)

        #Creating folders for image output during training.
        self.modeltraining_output               = self.Modeldir + "/training_output"
        self.modeltraining_output_images        = self.modeltraining_output + "/originals"
        self.modeltraining_output_corrections   = self.modeltraining_output + "/corrections"
        os.makedirs(self.modeltraining_output)
        os.makedirs(self.modeltraining_output_images)
        os.makedirs(self.modeltraining_output_corrections)

        self.Analytics_training("setup", 0, 0, 0, 0, 0, 0, 0, 0)
        self.Analytics_validation("setup", 0, 0, 0, 0, 0, 0, 0, 0)

    def Save_Model(self, epoch):
            if (epoch == 0) or (self.Generator_loss_validation[epoch] < np.min(self.Generator_loss_validation[:epoch])):
                torch.save(self.Generator.state_dict(), str( self.Modeldir + "/model.pt"))


    def SaveState(self):
        PSettings = list(self.Settings)
        stateloc = self.Modeldir + "/Savestate.txt"
        with open(stateloc, 'w') as f:
            for param in PSettings:
                f.write(param +  ": " + str(self.Settings[param]) + "\n")
            f.write("Generator model: " + self.Generator.name + "\n")
            f.write("Discriminator model: " + self.Discriminator.name + "\n")

    def CenteringAlgorithm(self, boxmult, Boxsize, SampleH, SampleW):
        """
        Returns new H W coordinates centered on the defect block
        """
        len = int(Boxsize * boxmult)
        len = int(np.floor( len + (Boxsize * 0.5) - (len * 0.5) ))

        return (SampleH - len), (SampleW - len)
        

    def Generator_updater(self, real_A, real_B, d_cord, val=False):       
        self.Discriminator.requires_grad=False

        self.Generator.zero_grad()
        
        # Generator loss
        fake_B = self.Generator(real_A)         
        predicted_fake = self.Discriminator(real_A, fake_B)
        loss_GAN = self.GAN_loss(predicted_fake, self.valid)
        
        #Pixelwise loss
        SampleH, SampleW, BoxSize = d_cord[0]
        SampleH, SampleW = self.CenteringAlgorithm(int(self.Settings["Loss_region_Box_mult"]), BoxSize, SampleH, SampleW)
        L1_loss_region = BoxSize * int(self.Settings["Loss_region_Box_mult"])
        loss_pixel = self.pixelwise_loss(fake_B, real_B)
        local_pixelloss = self.pixelwise_loss(fake_B[:,:,SampleH:SampleH+L1_loss_region,SampleW:SampleW+L1_loss_region], real_B[:,:,SampleH:SampleH+L1_loss_region,SampleW:SampleW+L1_loss_region])
        
        #Total loss
        Total_loss_Generator = loss_GAN + self.Settings["L1_loss_weight"] * loss_pixel + self.Settings["L1__local_loss_weight"] * local_pixelloss
        
        if not val:
            Total_loss_Generator.backward()
            self.Generator_optimizer.step()

        return Total_loss_Generator.item(), loss_pixel.item()

    def Discriminator_updater(self, real_A, real_B, val=False):
        self.Discriminator.requires_grad=True
        
        self.Discriminator.zero_grad()
        
        #Real loss
        predicted_real = self.Discriminator(real_A, real_B)
        
        loss_real = self.GAN_loss(predicted_real, self.valid)

        #Fake loss
        fake_B = self.Generator(real_A)
        predicted_fake = self.Discriminator(real_A, fake_B.detach())
        loss_fake = self.GAN_loss(predicted_fake, self.fake)

        #Total loss and backprop
        Total_loss_Discriminator = 0.5 * (loss_real + loss_fake)
        if not val: 
            Total_loss_Discriminator.backward() # backward run        
            self.Discriminator_optimizer.step() # step

        return Total_loss_Discriminator.item(), predicted_real, predicted_fake

    
    def FromTorchTraining(self, image):
        return image.mul(255).add_(0.5).clamp_(0,255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()


    def Generate_validation_images(self, epoch, real_A):
        self.Generator.eval()

        if real_A.size(0) > 1:
            real_im = real_A[0,:,:,:].clone()
        else:
            real_im = real_A.clone()

        with torch.no_grad():
            if real_A.size(0) > 1:
                fake_B = self.Generator(real_im.unsqueeze(0))
                im = Image.fromarray(self.FromTorchTraining(fake_B.squeeze(0)))
                co = Image.fromarray(self.FromTorchTraining(real_im))
            else:
                fake_B = self.Generator(real_im)
                im = Image.fromarray(self.FromTorchTraining(fake_B.squeeze(0)))
                co = Image.fromarray(self.FromTorchTraining(real_im.squeeze(0)))

            co.save(self.modeltraining_output_images + "/" + "Original_image_epoch_" + str(epoch) + ".jpg", "JPEG")
            im.save(self.modeltraining_output_corrections + "/" + "Generator_output_image_epoch_" + str(epoch) + ".jpg", "JPEG")

        self.Generator.train()

    def validation_run(self, val_loader, epoch):
        with torch.no_grad():
            current_GEN_loss = 0
            current_DIS_loss = 0
            Discrim_acc_real = 0
            Discrim_acc_fake = 0
            pixelloss = 0
            Discrim_acc_real_raw = 0
            Discrim_acc_fake_raw = 0
            real_fake_treshold = 5
            if self.Settings["batch_size"] == 1:
                val_unit = "image(s)"
            else:
                val_unit = "batch(s)"
            with tqdm(val_loader, unit=val_unit, leave=False) as tepoch:
                for images, defect_images, defect_coordinates in tepoch:
                    if epoch == 0:
                        tepoch.set_description(f"Validation run on Epoch {epoch}/{self.Settings['epochs']}")
                    elif epoch > 0:
                        tepoch.set_description(f"Validation Gen_loss {self.Generator_loss_validation[epoch-1]:.4f} Disc_loss {self.Discriminator_loss_validation[epoch-1]:.4f}")
                    self.valid = torch.ones((self.Settings["batch_size"], *self.patch), requires_grad=False).to(self.device)
                    self.fake = torch.zeros((self.Settings["batch_size"], *self.patch), requires_grad=False).to(self.device)

                    real_A = defect_images.to(self.device) #Defect
                    real_B = images.to(self.device) #Target 
                    defect_coordinates.to(self.device) # local loss coordinates

                    DIS_loss, predicted_real, predicted_fake = self.Discriminator_updater(real_A, real_B, val=True)
                    GEN_loss, loss_pixel = self.Generator_updater(real_A, real_B, defect_coordinates, val=True)

                    #Snapping image from generator during validation
                    if (epoch % 10) == 0:
                        self.Generate_validation_images(epoch, real_A)

                    #Analytics            
                    current_DIS_loss += DIS_loss  
                    current_GEN_loss += GEN_loss
                    pixelloss += loss_pixel
                    Discrim_acc_real += torch.sum(torch.sum(predicted_real, (2,3))/(self.patch[1]*2) > real_fake_treshold).item()
                    Discrim_acc_fake += torch.sum(torch.sum(predicted_fake, (2,3))/(self.patch[1]*2) < real_fake_treshold).item()
                    if self.Settings["batch_size"] == 1:
                        Discrim_acc_real_raw += torch.sum(predicted_real, (2,3)).item() / (self.patch[1]*2)
                        Discrim_acc_fake_raw += torch.sum(predicted_fake, (2,3)).item() / (self.patch[1]*2)
                    else:
                        Discrim_acc_real_raw += torch.sum(torch.sum(predicted_real, (2,3)) /(self.patch[1]*2), 0).item()
                        Discrim_acc_fake_raw += torch.sum(torch.sum(predicted_fake, (2,3)) /(self.patch[1]*2), 0).item()


            self.Analytics_validation(epoch, current_GEN_loss, current_DIS_loss, Discrim_acc_real, Discrim_acc_fake, Discrim_acc_real_raw, Discrim_acc_fake_raw, pixelloss, len(val_loader))

    def Trainer(self, train_loader, val_loader):
            epochs = tqdm(range(self.Settings["epochs"]), unit="epoch")
            for epoch in epochs:
                epochs.set_description(f"Training the model on epoch {epoch}")
                current_GEN_loss = 0
                current_DIS_loss = 0
                Discrim_acc_real = 0
                Discrim_acc_fake = 0
                pixelloss = 0
                Discrim_acc_real_raw = 0
                Discrim_acc_fake_raw = 0
                real_fake_treshold = 5
                if self.Settings["batch_size"] == 1:
                    tepoch = tqdm(train_loader, unit='image(s)', leave=False)
                else:
                    tepoch = tqdm(train_loader, unit='batch(s)', leave=False)

                for images, defect_images, defect_coordinates in tepoch:
                    if epoch == 0:
                        tepoch.set_description(f"Training on Epoch {epoch}/{self.Settings['epochs']}")
                    elif epoch > 0:
                        tepoch.set_description(f"Training Gen_loss {self.Generator_loss_train[epoch-1]:.4f} Disc_loss {self.Discriminator_loss_train[epoch-1]:.4f}")
                        
                    self.valid = torch.ones((self.Settings["batch_size"], *self.patch), requires_grad=False).to(self.device)
                    self.fake = torch.zeros((self.Settings["batch_size"], *self.patch), requires_grad=False).to(self.device)


                    real_A = defect_images.to(self.device) #Defect
                    real_B = images.to(self.device) #Target
                    defect_coordinates.to(self.device) # local loss coordinates
                    DIS_loss, predicted_real, predicted_fake = self.Discriminator_updater(real_A, real_B)
                    GEN_loss, loss_pixel, = self.Generator_updater(real_A, real_B, defect_coordinates)

                    #Analytics
                    current_GEN_loss += GEN_loss
                    current_DIS_loss += DIS_loss
                    pixelloss += loss_pixel                    
                    Discrim_acc_real += torch.sum(torch.sum(predicted_real, (2,3))/(self.patch[1]*2) > real_fake_treshold).item() 
                    Discrim_acc_fake += torch.sum(torch.sum(predicted_fake, (2,3))/(self.patch[1]*2) < real_fake_treshold).item() 
                    if self.Settings["batch_size"] == 1:
                        Discrim_acc_real_raw += torch.sum(predicted_real, (2,3)).item() /(self.patch[1]*2)
                        Discrim_acc_fake_raw += torch.sum(predicted_fake, (2,3)).item() /(self.patch[1]*2)
                    else:
                        Discrim_acc_real_raw += torch.sum(torch.sum(predicted_real, (2,3)) /(self.patch[1]*2), 0).item()
                        Discrim_acc_fake_raw += torch.sum(torch.sum(predicted_fake, (2,3)) /(self.patch[1]*2), 0).item()
                
                #Save per epoch
                self.Analytics_training(epoch, current_GEN_loss, current_DIS_loss, Discrim_acc_real, Discrim_acc_fake, Discrim_acc_real_raw, Discrim_acc_fake_raw, pixelloss, len(train_loader))
                self.validation_run(val_loader, epoch)
                self.Save_Model(epoch)
            #Save Analytics to file and create images from analytics    
            self.Save_Analytics()
            self.Create_graphs()
            self.SaveState()
            if self.transmit: # Send finished model to server storage
                print("Sending files")
                self.transmitter.send(self.Modeldir)
                self.transmitter.close()

    def Analytics_training(self, epoch, current_GEN_loss, current_DIS_loss, Discrim_acc_real, Discrim_acc_fake, Discrim_acc_real_raw, Discrim_acc_fake_raw, pixelloss, length):
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
            current_GEN_loss = current_GEN_loss / (length * self.Settings["batch_size"])
            current_DIS_loss = current_DIS_loss / (length * self.Settings["batch_size"])
            Discrim_acc_real = Discrim_acc_real / (length * self.Settings["batch_size"])
            Discrim_acc_fake = Discrim_acc_fake / (length * self.Settings["batch_size"])
            Discrim_acc_real_raw = Discrim_acc_real_raw / (length * self.Settings["batch_size"])
            Discrim_acc_fake_raw = Discrim_acc_fake_raw / (length * self.Settings["batch_size"])
            pixelloss = pixelloss / (length * self.Settings["batch_size"])

            self.Generator_loss_train[epoch] = current_GEN_loss
            self.Discriminator_loss_train[epoch] = current_DIS_loss
            self.Discriminator_accuracy_real_training[epoch] = Discrim_acc_real
            self.Discriminator_accuracy_fake_training[epoch] = Discrim_acc_fake
            self.Discriminator_accuracy_real_training_raw[epoch] = Discrim_acc_real_raw
            self.Discriminator_accuracy_fake_training_raw[epoch] = Discrim_acc_fake_raw
            self.Generator_pixel_loss_training[epoch] = pixelloss

    def Analytics_validation(self, epoch, current_GEN_loss, current_DIS_loss, Discrim_acc_real, Discrim_acc_fake, Discrim_acc_real_raw, Discrim_acc_fake_raw, pixelloss, length):
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
            current_GEN_loss = current_GEN_loss / (length * self.Settings["batch_size"])
            current_DIS_loss = current_DIS_loss / (length * self.Settings["batch_size"])
            Discrim_acc_real = Discrim_acc_real / (length * self.Settings["batch_size"])
            Discrim_acc_fake = Discrim_acc_fake / (length * self.Settings["batch_size"])
            Discrim_acc_real_raw = Discrim_acc_real_raw / (length * self.Settings["batch_size"])
            Discrim_acc_fake_raw = Discrim_acc_fake_raw / (length * self.Settings["batch_size"])
            pixelloss = pixelloss / (length * self.Settings["batch_size"])

            self.Generator_loss_validation[epoch] = current_GEN_loss
            self.Discriminator_loss_validation[epoch] = current_DIS_loss
            self.Discriminator_accuracy_real_validation[epoch] = Discrim_acc_real 
            self.Discriminator_accuracy_fake_validation[epoch] = Discrim_acc_fake
            self.Discriminator_accuracy_real_validation_raw[epoch] = Discrim_acc_real_raw
            self.Discriminator_accuracy_fake_validation_raw[epoch] = Discrim_acc_real_raw
            self.Generator_pixel_loss_validation[epoch] = pixelloss

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
        plt.title("Generator pixel loss")
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
    def __init__(self, modelref, dataloader, device="cpu"):
        self.model = modelref
        self.device = device
        self.transform = transforms.ToPILImage()
        self.dataloader = dataloader
        self.models_loc = "Trained_Models"
        self.Inference_dir = "Inference_Run"
        models = os.listdir(self.models_loc)

        for num, model in enumerate(models):
            choice = "[" + str(num) + "]    " + model
            print(choice)

        choice  = int(input("please input modelnum: "))

        self.modeldir = self.models_loc + "/"  + models[choice] + "/model.pt"
        self.modelname = models[choice]
        self.run_dir = self.Inference_dir + "/" + self.modelname
        os.makedirs(self.run_dir)
        os.makedirs(self.run_dir + "/original")
        os.makedirs(self.run_dir + "/reconstruction")
        self.RestoreModel()
        self.Inference_run()

    def RestoreModel(self):
        self.model.load_state_dict(torch.load(self.modeldir, map_location=torch.device(self.device)))
        print("Succesfully loaded", self.modelname, "model")

    def FromTorchTraining(self, image):
        return image.mul(255).add_(0.5).clamp_(0,255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        
    def Inference_run(self, runs=3):
        """
        Does an inference run on the Model for three images
        """
        self.model.eval()
        loader = tqdm(range(runs))
        with torch.no_grad():
            for run in loader:
                loader.set_description(f"Running {run}/{runs} images completed")
                _ , image, _ = next(iter(self.dataloader))

                fake_B = self.model(image)
                im = Image.fromarray(self.FromTorchTraining(fake_B.squeeze(0)))
                co = Image.fromarray(self.FromTorchTraining(image.squeeze(0)))
                co.save(self.run_dir + "/original/original_image" + str(run) + ".jpg")
                im.save(self.run_dir + "/reconstruction/reconstructed_image" + str(run) + ".jpg")
        print("Done!")
        print("All results saved to:")
        print(self.run_dir)
if __name__ == '__main__':
    pass