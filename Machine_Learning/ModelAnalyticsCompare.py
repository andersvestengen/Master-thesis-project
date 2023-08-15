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
    "self.Generator_loss_validation",  # WGAN
    "self.Discriminator_loss_validation",  #WGAN
    "self.Discriminator_accuracy_real_validation_raw", #WGAN
    "self.Discriminator_accuracy_fake_validation_raw", #WGHA
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


def MakeSaveGraph(datas, xlabel, ylabel, title):
    for arr in datas:
        if len(arr) == 3:
            xaxis, axis, legend = arr
        else:
            xaxis = torch.arange(axis.size(0))
        plt.plot(xaxis, axis, label=legend)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
    #plt.savefig("Images" + "/" + title + ".png")
    #plt.clf()

def ChooseAnalytics():
    choice = input("Old or new models [o/n]?: ")
    analytic = []
    if choice == "o":
        ListPrint(Old_list)
        while True:
            choice = input("which [num] analytic? (x to stop): ")
            if choice == "x" or choice == "X":
                break
            else:
                analytic.append(int(choice))
    if choice == "n":
        ListPrint(New_list)
        while True:
            choice = input("which [num] analytic? (x to stop): ")
            if choice == "x" or choice == "X":
                break
            else:
                analytic.append(int(choice))
    return analytic

def GetData(Models, analytics_dim, gen):
    Datas = []
    if gen == "old":
                for num, modeldir in enumerate(Models):
                     for dim in analytics_dim:
                        data = np.load(modeldir + "/Analytics.npz")
                        Datas.insert(num, data['arr_0'][dim])
    else:
                for num, modeldir in enumerate(Models):
                     for dim in analytics_dim:
                        data = torch.load(modeldir + "/Analytics.pt", map_location=torch.device('cpu'))
                        Datas.insert(num, data[0][dim])
    return Datas

def LoadModelGraphs(Models):
    old = 0
    new = 0
    for modeldir in Models:
        print("Modelname: ", modeldir.split("/")[1].split(" ")[0])
        for name in os.listdir(modeldir):
            if name == "Analytics.npz":
                old += 1
            if name == "Analytics.pt":
                new  += 1
    if new != 0 and old != 0:
        print("model choice include models using mixed formats. Either select pre- or post- July 19th!")
        for modeldir in Models:
            print(modeldir)
            sys.exit()
    
    analytic_dim = ChooseAnalytics() # list of the different analytics to compare
    if old != 0: # If old 
        models_data = GetData(Models, analytic_dim, gen="old") #Returns the data in (model,dim) hierarchy
    if new != 0: 
        models_data = GetData(Models, analytic_dim, gen="new") #Returns the data in (model,dim) hierarchy

    Graph_list = []
    print("This is the models_data list size:", len(models_data[0])) #So the list is just len = 2 ? 
    for modeldata in models_data:
        modelname = modeldir.split("/")[1].split(" ")[0]
        for data in modeldata:
              Graph_list.append([data, modelname])

    MakeSaveGraph(Graph_list, "input", "output", "Model Comparison")


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
