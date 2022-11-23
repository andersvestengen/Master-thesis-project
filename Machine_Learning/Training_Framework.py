import paramiko
import os
from datetime import datetime
"""
TODO: GOAL: Create class for holding functions and values associated with training the ML models
        SubGoal: I want to be able to view analytics.npz image, and send model files to other server.

    - Create Initial class
    - Create Function which can create Images from Analytics file
    - Create function which can send trained models along with their analytics to other storage.
    - Create Function which can hold SSH username and password for file transfer. 

"""
class FileSender():
    def __init__(self):
        self.externalroot = "dummytransferdir/"
        print("setting up ssh and sftp")
        self.username = input("input username: ")
        self.server = "login.uio.no"
        self.password = input("input password: ")

        self.cli = paramiko.SSHClient()
        self.cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.cli.connect(hostname=self.server, port=22, username=self.username, password=self.password)
        print("Succesfully connecte to", self.server)
        self.ftr = self.cli.open_sftp()
        print("sftp open")
    

    def send(self, name, location):
        self.ftr.put(location, self.externalroot + name)

    def close(self):
        self.ftr.close()
        self.cli.close()

class Training_Framework():
    def __init__(self, Settings):
        self.Settings = Settings
        # Set the working Directory
        if not self.Settings["workingdir"]:
            self.workingdir = os.getcwd()
        else:
            self.workingdir = self.Settings["workingdir"]

        time = str(datetime.now())
        if not Settings["ModelName"]:
            self.Modeldir = "GAN_Model_" + self.Settings["epochs"] + "_" + self.Settings["batch_size"] + time[:-7].strip(" ")
            
        os.mkdir(self.Modelname)

    def make_loss_img(self):
        pass
