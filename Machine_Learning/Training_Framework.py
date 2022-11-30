import paramiko
import os
from datetime import datetime
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

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
        for filename in dir_struct[0][2]:
            file_external_path = self.externaldir + "/" + foldername + "/" + filename
            file_local_path = dir_struct[0][0] + "/" + filename
            self.ftr.put(file_local_path ,file_external_path)
        print("finished sending directory", foldername)


    def pull(self, directory):
        dir_struct = self.ftr.listdir(self.externaldir + "/" + directory)
        os.mkdir(self.local_Model_Directory + "/" + directory)
        for filename in dir_struct:
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

    def send_tester(self):
        # Make Directories
        print("starting test of filesending infrastructure!")
        dirs = []
        workingdir = self.local_Model_Directory
        for dir in range(5):
            dirname = workingdir + "/" "Testdir" + str(dir)
            dirs.append(dirname)
            os.mkdir(dirname)
            for file in range(5):
                filename = dirname + "/File" + str(file) + ".txt"
                with open(filename, 'w') as f:
                    f.write("Im a new textfile!")
        
        #Send those files to the server!
        print("Starting filetransfer!")
        for directory in dirs:
            self.send(directory)




class Training_Framework():

    """
    Framework for training the different networks, should not inherit or mess with pytorch itself, but instead passes the model by assignment to make training more like 
    Legos and less like a novel.
    

    TODO:
        - import the different methods of training as differing functions.
        - then create the larger batch-training etc. functions.
        - Bottom up architecture. 

        - Function to keep track of the analytics 
        - function to save figures of the analytics
        - Add accuracy of the discriminator to the analytics (maybe in its own window.)
    """
    def __init__(self, Settings):
        self.Settings = Settings
        
        # Set the working Directory
        if not self.Settings["workingdir"]:
            self.workingdir = os.getcwd()
        else:
            self.workingdir = self.Settings["workingdir"]

        #Create the directory of the model (Look back at this for batch training.)
        time = str(datetime.now())
        if Settings["ModelName"] is None:
            stamp = time[:-16] + "_" + time[11:-7]
            self.Modeldir = Settings["workingdir"] + "/Trained_Models/" + "GAN_Model_" + self.Settings["epochs"] + "_" + self.Settings["batch_size"] + "_" + self.Settings["lr"] + "_" + stamp
            
        os.mkdir(self.Modeldir)


    def Analytics_training(self, *args):
        """
        current epoch needs to be the first argument, except when setting up training.  
        """
        if args[0] == "setup":
            self.Generator_loss_train = np.zeros(self.Settings["epochs"])
            self.Discriminator_loss_train = np.zeros(self.Settings["epochs"])
            self.Discriminator_accuracy_training = np.zeros(self.Settings["epochs"])


        else:
            epoch = args[0]
            self.Generator_loss_train[epoch] = args[1]
            self.Discriminator_loss_train[epoch] = args[2]
            self.Discriminator_accuracy_training[epoch] = args[3]


    def Analytics_validation(self, *args):
        """
        current epoch needs to be the first argument, except when setting up training. 
        """
        if args[0] == "setup":
            self.Generator_loss_validation = np.zeros(self.Settings["epochs"])
            self.Discriminator_loss_validation = np.zeros(self.Settings["epochs"])
            self.Discriminator_accuracy_validation = np.zeros(self.Settings["epochs"])

        else:
            epoch = args[0]
            self.Generator_loss_validation[epoch] = args[1]
            self.Discriminator_loss_validation[epoch] = args[2]
            self.Discriminator_accuracy_validation[epoch] = args[3]  


    def Save_Analytics(self):
        np.savez(self.Modeldir, (self.Generator_loss_validation,
                                self.Discriminator_loss_validation,
                                self.Discriminator_accuracy_validation,
                                self.Generator_loss_train,
                                self.Discriminator_loss_train,
                                self.Discriminator_accuracy_training
                                ))

    def trainer(self):
        """
        For now will only train one model at a time, but should be remade to train batches of models. 
        """

    def Train_init(self, Generator, Discriminator):
        pass



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

    def Create_graphs(self):
        decision = input("show[y] or just save[n]? [y/n]: ")
        loaded_arrays = np.load(self.modeldir + "/" + "Analytics.npz")
        xaxis = np.arange(0, loaded_arrays['arr_0'].shape[0])
        plt.plot(xaxis, loaded_arrays['arr_0'], label="Generator loss training")
        plt.plot(xaxis, loaded_arrays['arr_1'], label="Generator loss validation")
        plt.plot(xaxis, loaded_arrays['arr_2'], label="Discriminator loss training")    
        plt.plot(xaxis, loaded_arrays['arr_3'], label="Discriminator loss validation")
        plt.xlabel("epochs")
        plt.ylabel("Loss")
        plt.title("Model loss curves")
        plt.legend()
        if decision == "y":
            plt.savefig(self.modeldir + "/" + "model_loss_curves.png")
            plt.show()
        else:
            plt.savefig()

        plt.clf() # clear the plot
        plt.plot(xaxis, loaded_arrays['arr_4'], label="Discriminator accuracy training")
        plt.plot(xaxis, loaded_arrays['arr_5'], label="Discriminator accuracy validation")
        plt.xlabel("epochs")
        plt.ylabel("Percentage [%]")
        plt.title("Discriminator accuracy")
        plt.legend()
        if decision == "y":
            plt.savefig(self.modeldir + "/" + "discriminator_accuracy_curves.png")
            plt.show()
        else:
            plt.savefig()
        plt.clf() # clear the plot

if __name__ == '__main__':
    pass