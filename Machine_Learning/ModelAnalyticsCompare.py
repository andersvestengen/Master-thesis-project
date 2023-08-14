import torch
import os
import numpy as np
import matplotlib.pyplot as plt 
import sys
"""
File Situation:

New Structure: (July 19th Onwards ) in Pytorch:

self.Generator_loss_validation, 
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

Old Structure in Numpy!:

self.Generator_loss_validation, 
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

"""

New_list = [
    "self.Generator_loss_validation", 
    "self.Discriminator_loss_validation", 
    "self.Discriminator_accuracy_real_validation_raw", 
    "self.Discriminator_accuracy_fake_validation_raw", 
    "self.Discriminator_auto_loss_training", 
    "self.Generator_pixel_loss_validation", 
    "self.Generator_local_pixel_loss_validation", 
    "self.Generator_DeepFeatureLoss_validation", 
    "self.Generator_DeepFeatureLoss_training", 
    "self.Generator_loss_train", 
    "self.Generator_pixel_loss_training", 
    "self.Generator_local_pixel_loss_training", 
    "self.Discriminator_loss_train", 
    "self.Discriminator_accuracy_real_training_raw", 
    "self.Discriminator_accuracy_fake_training_raw", 
    "self.Discriminator_auto_loss_validation", 
    "self.Generator_auto_loss_validation", 
    "self.PSNR_Generated", 
    "self.PSNR_Generated_patch", 
    "self.SSIM_Generated", 
    "self.SSIM_Generated_patch",]

Old_list = [
    "self.Generator_loss_validation", 
    "self.Discriminator_loss_validation", 
    "self.Discriminator_accuracy_real_validation_raw", 
    "self.Discriminator_accuracy_fake_validation_raw", 
    "self.Generator_pixel_loss_validation", 
    "self.Generator_local_pixel_loss_validation", 
    "self.Generator_DeepFeatureLoss_validation",
    "self.Generator_DeepFeatureLoss_training", 
    "self.Generator_loss_train", 
    "self.Generator_pixel_loss_training", 
    "self.Generator_local_pixel_loss_training", 
    "self.Discriminator_loss_train", 
    "self.Discriminator_accuracy_real_training_raw", 
    "self.Discriminator_accuracy_fake_training_raw", 
    "self.Discriminator_auto_loss_validation", 
    "self.Generator_auto_loss_validation",]

def ListPrint(list):
    for num, item in enumerate(list):
        choice = "[" + str(num) + "]    " + item
        print(choice)


def MakeSaveGraph(datas, xlabel, ylabel, title, saveloc):
    for axis, legend in datas:
        xaxis = torch.arange(axis.size(0))
        plt.plot(xaxis, axis, label=legend)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(saveloc + ".png")
    plt.clf()

def ChooseAnalytics():
    choice = input("Old or new models [o/n]?: ")
    analytic = []
    if choice == "o":
        ListPrint(Old_list)
        while True:
            choice = int(input("which [num] analytic? (x to stop): "))
            if choice == "x" or choice == "X":
                break
            else:
                analytic.append(choice)
    if choice == "n":
        ListPrint(New_list)
        while True:
            choice = int(input("which [num] analytic? (x to stop): "))
            if choice == "x" or choice == "X":
                break
            else:
                analytic.append(choice)
    return analytic


def LoadModelGraphs(Models):
    old = 0
    new = 0
    for modeldir in Models:
        print("Modelname: ", modeldir.split("/")[1].split(" ")[0])
        for name in os.listdir(modeldir):
            if name == "Analytics.npz":
                old += 1
                print("Model is old!")
                data = np.load(modeldir + "/Analytics.npz")
                print("Data shape:", data['arr_0'].shape)
            if name == "Analytics.pt":
                new  += 1
                print("Model is new!")
                data = torch.load(modeldir + "/Analytics.pt", map_location=torch.device('cpu'))
                print(data[0].size())

    if new != 0 and old != 0:
        print("model choice include models using mixed formats. Either select pre- or post- July 19th!")
        for modeldir in Models:
            print(modeldir)
            sys.exit()
    
    analytic_dim = ChooseAnalytics
    if old != 0: # If old 
        models_data = []
        for dim in analytic_dim:
            pass

    


models_loc = "Trained_Models"
Inference_dir = "Inference_Run"
models = os.listdir(models_loc)

ListPrint(models)
modelchoices = []
while True:
    choice  = input("please input modelnum (x to stop): ")
    if choice == "x" or choice == "X":
        break
    else:
        modelchoices.append(models_loc + "/"  + models[int(choice)])

LoadModelGraphs(modelchoices)
ChooseAnalytics()






"""

model_state = models_loc + "/"  + models[choice] + "/Savestate.txt"
model_inf = []
with open(model_state, 'r') as f:
    model_inf = [Line for Line in f]
"""
