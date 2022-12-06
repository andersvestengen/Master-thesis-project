import paramiko
import os
from datetime import datetime
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

class FileSender():
    """
    This class sets up sending files between the local directory given (!Only expects no subfolders!) and the uio folder "Master_Thesis_Model_Directory/"
    It will make it easier to work with as the training cluster is not accessible to IP's outside the uio servers.

    *In the future maybe add some functionality to pull from the uio server to the local folder where this program is run.
    
    """
    def __init__(self):
        self.externaldir = "Master_Thesis_Model_Directory"
        print("setting up ssh and sftp")
        self.username = input("input username: ")
        self.server = "login.uio.no"
        self.password = input("input password: ")
        
        self.cli = paramiko.SSHClient()
        self.cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.cli.connect(hostname=self.server, port=22, username=self.username, password=self.password)
        print("Succesfully connected to", self.server)
        self.ftr = self.cli.open_sftp()
        print("sftp open")
        self.local_Model_Directory = "Trained_Models"


    def send(self, directory):
        dir_struct = list(os.walk(directory))
        foldername = dir_struct[0][0].split("/")[-1]
        self.ftr.mkdir(self.externaldir + "/" + foldername)
        for filename in tqdm(dir_struct[0][2], unit="file", desc="Sending {foldername} to server"):
            file_external_path = self.externaldir + "/" + foldername + "/" + filename
            file_local_path = dir_struct[0][0] + "/" + filename
            self.ftr.put(file_local_path ,file_external_path)
        print("finished sending directory", foldername)


    def pull(self, directory):
        dir_struct = self.ftr.listdir(self.externaldir + "/" + directory)
        os.makedirs(self.local_Model_Directory + "/" + directory)
        for filename in tqdm(dir_struct, unit="file", desc="downloading folder {directory}"):
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
        self.Settings = Settings
        self.Generator = Generator
        self.Generator_optimizer = G_opt
        self.GAN_loss = GAN_loss
        self.pixelwise_loss = pixelwise_loss
        self.Discriminator = Discriminator
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
        if self.Settings["ModelName"] is not None:
            self.Modeldir = self.workingdir +  "/Trained_Models/" + "GAN_Model_" + self.Settings["ModelName"] + "_time-" + stamp
        else:
            self.Modeldir = self.workingdir + "/Trained_Models/" + "GAN_Model" + "_time-" + stamp
            
        os.makedirs(self.Modeldir)

    def Save_Model(self, epoch):
            if (epoch == 0) or (self.Generator_loss_validation[epoch] < self.Generator_loss_validation[epoch-1]):
                torch.save(self.Generator.state_dict(), str( self.Modeldir + "/model.pt"))

    def Generator_updater(self, real_A, real_B, fake_B, val=False):
        self.Generator_optimizer.zero_grad()
        
        # Generator loss            
        predict_fake = self.Discriminator(fake_B, real_A) # Compare fake output to original image
        loss_GAN = self.GAN_loss(predict_fake, self.valid)
        #Pixelwise loss
        loss_pixel = self.pixelwise_loss(fake_B, real_B) # might be misinterpreting the loss inputs here.
        
        #Total loss
        Total_loss_Generator = loss_GAN + self.Settings["L1_loss_weight"] * loss_pixel
        
        if not val:
            Total_loss_Generator.backward()
            self.Generator_optimizer.step()

        return Total_loss_Generator.item()

    def Discriminator_updater(self , predicted_real, predicted_fake, val=False):
        self.Discriminator_optimizer.zero_grad()
        
        # Real loss 
        loss_real = self.GAN_loss(predicted_real, self.valid)
        
        # Fake loss
        loss_fake = self.GAN_loss(predicted_fake, self.fake)
        # Total loss
        Total_loss_Discriminator = 0.5 * (loss_real + loss_fake)
        if not val:
            Total_loss_Discriminator.backward()
            
            self.Discriminator_optimizer.step()

        return Total_loss_Discriminator.item()

    def Discriminator_updater_real_fake_separate(self, predicted_real, predicted_fake, val=False):
        self.Discriminator_optimizer.zero_grad()
        
        # Real loss 
        loss_real = self.GAN_loss(predicted_real, self.valid)
        
        # Fake loss
        loss_fake = self.GAN_loss(predicted_fake, self.fake)
        Total_loss_Discriminator = 0.5 * (loss_real + loss_fake)
        if not val:
            Total_loss_Discriminator.backward()
            self.Discriminator_optimizer.step()

        return (loss_real.item() + loss_fake.item())*0.5


    def Discriminator_updater_staggered(self, predicted_real, predicted_fake, epoch, val=False):
        self.Discriminator_optimizer.zero_grad()
        
        # Real loss 
        loss_real = self.GAN_loss(predicted_real, self.valid)
        
        loss_fake = self.GAN_loss(predicted_fake, self.fake)

        Total_loss_Discriminator = 0.5 * (loss_real + loss_fake)
        if not val and epoch > 50:

            Total_loss_Discriminator.backward()
            self.Discriminator_optimizer.step()

        return (loss_real.item() + loss_fake.item())*0.5

    def validation_run(self, val_loader, epoch):
            current_GEN_loss = 0
            current_DIS_loss = 0
            Discrim_acc_real = 0
            Discrim_acc_fake = 0
            with tqdm(val_loader, unit='batch', leave=False) as tepoch:
                for image, defect_images in tepoch:
                    tepoch.set_description(f"Validation run on Epoch {epoch}/{self.Settings['epochs']}")
                    if epoch > 0:
                        tepoch.set_description(f"Validation Gen_loss {self.Generator_loss_validation[epoch-1]:.5f} Disc_loss {self.Discriminator_loss_validation[epoch-1]:.5f}")
  
                    self.valid = torch.ones((self.Settings["batch_size"], *self.patch)).to(self.device)
                    self.fake = torch.zeros((self.Settings["batch_size"], *self.patch)).to(self.device)

                    real_A = defect_images.to(self.device)
                    real_B = image.to(self.device)
                    
                    fake_B = self.Generator(real_A)
                    current_GEN_loss += self.Generator_updater(real_A, real_B, fake_B, val=True) / self.Settings["batch_size"]
                    predicted_real = self.Discriminator(real_B, real_A)
                    predicted_fake = self.Discriminator(fake_B.detach(), real_A)
                    current_DIS_loss += self.Discriminator_updater_staggered(predicted_real, predicted_fake, epoch, val=True) / self.Settings["batch_size"]
                    Discrim_acc_real += torch.sum(torch.sum(predicted_real, (2,3))/self.patch[1] > 1) / self.Settings["batch_size"]
                    Discrim_acc_fake += torch.sum(torch.sum(predicted_fake, (2,3))/self.patch[1] < 1) / self.Settings["batch_size"]

            current_GEN_loss = current_GEN_loss / len(val_loader)
            current_DIS_loss = current_DIS_loss / len(val_loader)
            Discrim_acc_real = Discrim_acc_real / len(val_loader)
            Discrim_acc_fake = Discrim_acc_fake / len(val_loader)
            if epoch == 0:
                self.Analytics_validation("setup", current_GEN_loss, current_DIS_loss, Discrim_acc_real, Discrim_acc_fake)
            else:
                self.Analytics_validation(epoch, current_GEN_loss, current_DIS_loss, Discrim_acc_real, Discrim_acc_fake)

    def Trainer(self, train_loader, val_loader):
            for epoch in tqdm(range(self.Settings["epochs"]), unit="epoch", desc="Training the model on epoch {epoch}"):
                current_GEN_loss = 0
                current_DIS_loss = 0
                Discrim_acc_real = 0
                Discrim_acc_fake = 0
                with tqdm(train_loader, unit='batch', leave=False) as tepoch:
                    for images, defect_images in tepoch:
                        tepoch.set_description(f"Training on Epoch {epoch}/{self.Settings['epochs']}")
                        if epoch > 0:
                            tepoch.set_description(f"Training Gen_loss {self.Generator_loss_train[epoch-1]:.5f} Disc_loss {self.Discriminator_loss_train[epoch-1]:.5f}")
                            
                        self.valid = torch.ones((self.Settings["batch_size"], *self.patch)).to(self.device)
                        self.fake = torch.zeros((self.Settings["batch_size"], *self.patch)).to(self.device)

                        real_A = defect_images.to(self.device)
                        real_B = images.to(self.device)
                        
                        fake_B = self.Generator(real_A)
                        current_GEN_loss += self.Generator_updater(real_A, real_B, fake_B) / self.Settings["batch_size"]
                        predicted_real = self.Discriminator(real_B, real_A)
                        predicted_fake = self.Discriminator(fake_B.detach(), real_A)
                        current_DIS_loss += self.Discriminator_updater_staggered(predicted_real, predicted_fake, epoch) / self.Settings["batch_size"]
                        Discrim_acc_real += torch.sum(torch.sum(predicted_real, (2,3))/self.patch[1] > 1) / self.Settings["batch_size"]
                        Discrim_acc_fake += torch.sum(torch.sum(predicted_fake, (2,3))/self.patch[1] < 1) / self.Settings["batch_size"]

                current_GEN_loss = current_GEN_loss / len(train_loader)
                current_DIS_loss = current_DIS_loss / len(train_loader)
                Discrim_acc_real = Discrim_acc_real / len(train_loader)
                Discrim_acc_fake = Discrim_acc_fake / len(train_loader)
                if epoch == 0:
                    self.Analytics_training("setup", current_GEN_loss, current_DIS_loss, Discrim_acc_real, Discrim_acc_fake)
                else:
                    self.Analytics_training(epoch, current_GEN_loss, current_DIS_loss, Discrim_acc_real, Discrim_acc_fake)
                self.validation_run(val_loader, epoch)
                self.Save_Model(epoch)
            self.Save_Analytics()
            self.Create_graphs()
            if self.transmit:
                print("Sending files")
                self.transmitter.send(self.Modeldir)
                self.transmitter.close()

    def Analytics_training(self, *args):
        """
        current epoch needs to be the first argument, except when setting up training.  
        """
        if args[0] == "setup":
            self.Generator_loss_train = np.zeros(self.Settings["epochs"])
            self.Discriminator_loss_train = np.zeros(self.Settings["epochs"])
            self.Discriminator_accuracy_real_training = np.zeros(self.Settings["epochs"])
            self.Discriminator_accuracy_fake_training = np.zeros(self.Settings["epochs"])


        else:
            epoch = args[0]
            self.Generator_loss_train[epoch] = args[1]
            self.Discriminator_loss_train[epoch] = args[2]
            self.Discriminator_accuracy_real_training[epoch] = args[3]
            self.Discriminator_accuracy_fake_training[epoch] = args[4]


    def Analytics_validation(self, *args):
        """
        current epoch needs to be the first argument, except when setting up training. 
        """
        if args[0] == "setup":
            self.Generator_loss_validation = np.zeros(self.Settings["epochs"])
            self.Discriminator_loss_validation = np.zeros(self.Settings["epochs"])
            self.Discriminator_accuracy_real_validation = np.zeros(self.Settings["epochs"])
            self.Discriminator_accuracy_fake_validation = np.zeros(self.Settings["epochs"])

        else:
            epoch = args[0]
            self.Generator_loss_validation[epoch] = args[1]
            self.Discriminator_loss_validation[epoch] = args[2]
            self.Discriminator_accuracy_real_validation[epoch] = args[3]  
            self.Discriminator_accuracy_fake_validation[epoch] = args[4]  


    def Save_Analytics(self):
        np.savez(self.Modeldir + '/Analytics.npz', (self.Generator_loss_validation,
                                self.Discriminator_loss_validation,
                                self.Discriminator_accuracy_real_validation,
                                self.Discriminator_accuracy_fake_validation,
                                self.Generator_loss_train,
                                self.Discriminator_loss_train,
                                self.Discriminator_accuracy_real_training,
                                self.Discriminator_accuracy_fake_training
                                ))

    def Create_graphs(self):
        xaxis = np.arange(1, self.Generator_loss_validation.shape[0]+1)
        plt.plot(xaxis, self.Generator_loss_train, label="Generator loss training")
        plt.plot(xaxis, self.Generator_loss_validation, label="Generator loss validation")
        plt.plot(xaxis, self.Discriminator_loss_train, label="Discriminator loss training")    
        plt.plot(xaxis, self.Discriminator_loss_validation, label="Discriminator loss validation")
        plt.xlabel("epochs")
        plt.ylabel("Loss")
        plt.title("Model loss curves")
        plt.legend()
        plt.savefig(self.Modeldir + "/model_loss_curves.png")
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



class Model_Inference():
    """
    Class to do inference and get inference data from the model.
    Made to only do inference on the Generator part of the GAN network.

    Input:
        - Modelref: This is the vanilla model class itself, from the Models/ dir
        - Modeldir: This is the location of the trained model.pt file.
    """
    def __init__(self, modelref, modeldir, device="cpu"):
        self.model = modelref
        self.device = device
        self.transform = transforms.ToPILImage()
        self.modeldir = modeldir
        self.modelname = self.modeldir.split("/")[-2]


    def RestoreModel(self):
        self.model.load_state_dict(torch.load(self.modeldir, map_location=torch.device(self.device)))
        print("Succesfully loaded", self.modelname, "model") # Make this reference the model name. 

    def GAN_Dataset_1_init():
        pass
        # Create a routine for extracting images from the GAN_1_dataset here.

    def Inference_run(self, image):
        """
        Does an inference run on the Model for a single image, requires an image from the Dataset.
        """
        print("Would you like to save or view images?")
        decision = input("type save/view: ")
        self.model.eval()
        Generated_output_image = self.model(image)
        im = self.transform(image.squeeze(0))
        output = self.transforms(Generated_output_image.squeeze(0))
        if decision == "view":
            f, (ax1,ax2) = plt.subplot(1,2)
            ax1.imshow(np.asarray(im))
            ax2.imshow(np.asarray(output))
            plt.show()
        if decision == "save":
            im.save(self.modelname + "defect_sample_input.jpg")
            output.save(self.modelname + "reconstructed_image.jpg")

if __name__ == '__main__':
    pass