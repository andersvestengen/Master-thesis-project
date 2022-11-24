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
        self.externaldir = "dummytransferdir/"
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
        print("received directory", directory)
        foldername = dir_struct[0][0]
        self.ftr.mkdir(self.externaldir + "/" + foldername)
        print("made remote directory", self.externaldir + "/" + foldername)
        for filename in dir_struct[0][2]:
            print("found file", filename, "in", foldername)
            file_external_path = self.externaldir + "/" + foldername + "/" + filename
            file_local_path = foldername + "/" + filename
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
            dirname = workingdir + str(dir)
            dirs.append(dirname)
            os.makedir(dirname)
            for file in range(5):
                filename = dirname + "/File" + str(file) + ".txt"
                with open(filename, 'w') as f:
                    f.write("Im a new textfile!")
        
        #Send those files to the server!
        print("Starting filetransfer!")
        for directory in dirs:
            self.send(directory)





class Training_Framework():
    def __init__(self, Settings):
        self.Settings = Settings



        # Do I want to train or do Inference?


        
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


if __name__ == '__main__':
    Sender = FileSender()
    Sender.send_tester()