import paramiko
import os
from datetime import datetime
import numpy as np
"""
TODO: GOAL: Create class for holding functions and values associated with training the ML models
        SubGoal: I want to be able to view analytics.npz image, and send model files to other server.

    - Create Initial class
    - Create Function which can create Images from Analytics file
    - Create function which can send trained models along with their analytics to other storage.
    - Create Function which can hold SSH username and password for file transfer. 

"""
class FileSender():
    """
    This class sets up sending files between the local directory given (!Only expects no subfolders!) and the uio folder "Master_Thesis_Model_Directory/"
    It will make it easier to work with as the training cluster is not accessible to IP's outside the uio servers.

    *In the future maybe add some functionality to pull from the uio server to the local folder where this program is run.
    
    """
    def __init__(self):
        self.externaldir = "Master_Thesis_Model_Directory/"
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
    

    def send(self, directory):
        dir_struct = list(os.walk(directory))
        foldername = dir_struct[0][0].split("/")[-1]
        self.ftr.mkdir(self.externaldir + "/" + foldername)
        for filename in dir_struct[0][2]:
            file_external_path = self.externaldir + "/" + foldername + "/" + filename
            file_local_path = dir_struct[0][0] + "/" + filename
            self.ftr.put(file_local_path ,file_external_path)
            print("sent", filename, "to new path", file_external_path)
        print("finished sending directory", foldername)


    def close(self):
        self.ftr.close()
        self.cli.close()

    def send_tester(self):
        # Make Directories
        print("starting test of filesending infrastructure!")
        dirs = []
        workingdir = "Trained_Models"
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


        """
        #Create the directory of the model (Look back at this for batch training.)
        time = str(datetime.now())
        if not Settings["ModelName"]:
            self.Modeldir = "GAN_Model_" + self.Settings["epochs"] + "_" + self.Settings["batch_size"] + time[:-7].strip(" ")
            
        os.mkdir(self.Modeldir)
        """

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



if __name__ == '__main__':
    pass