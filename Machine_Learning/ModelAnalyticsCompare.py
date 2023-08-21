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
    """
    Prints input to screen, line by line.
    """
    for num, item in enumerate(list):
        choice = "[" + str(num) + "]    " + item
        print(choice)


def MakeSaveGraph(datas, xlabel, ylabel, title):
    """
    Creates a graph from a variable range of input data
    
    """
    for arr in datas:
        if len(arr) == 3:
            xaxis, axis, legend = arr
        else:
            axis, legend = arr
            xaxis = torch.arange(axis.size(0))
        plt.plot(xaxis, axis, label=legend)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig("ModelCompareImages" + "/" + title + ".png")
    plt.clf()

def GetDataAlg(Models, Analytics, gen):
    Struct = []
    for model in Models:
        Modeldata = []
        for dim in Analytics:
            if gen == "old":
                data = np.load(model + "/Analytics.npz")
                data_dim = data['arr_0'][dim]
            if gen == "new":
                data = torch.load(model + "/Analytics.pt", map_location=torch.device('cpu'))
                if dim <= 16:
                    data_dim = data[dim][10:]
                else:
                    data_dim = data[dim]
            Modeldata.append(data_dim)
        Struct.append(Modeldata)
    return Struct


def ChooseAnalytics(filesys):
    analytic = []
    if filesys == "old":
        ListPrint(Old_list)
        while True:
            choice = input("which [num] analytic? (blank space to stop): ")
            if choice == "":
                break
            else:
                analytic.append(int(choice))
    if filesys == "new":
        ListPrint(New_list)
        while True:
            choice = input("which [num] analytic? (blank space to stop)): ")
            if choice == "":
                break
            else:
                analytic.append(int(choice))
    return analytic

def DisplayGraphs(Analytics, Struct, Models, gen):
    if gen == "old":
        anylytics = Old_list
    else:
        anylytics = New_list
    smoothing = input("apply smoothing? [y/n]: ")
    for n in range(len(Analytics)):
        GraphData = []
        print("Analytic is:", anylytics[Analytics[n]])
        for m, model in enumerate(Models):
            if smoothing == "y":
                Gdata = SmoothCurve(Struct[m][n])
            else:
                Gdata = Struct[m][n]
            modelname = model.split("/")[1].split(" ")[0]
            GraphData.append([Gdata, modelname])
        xlabel = input("what is xlabel?: ")
        ylabel = input("what is ylabel?: ")
        title = input("what is title?: ")
        MakeSaveGraph(GraphData, xlabel, ylabel, title)


def SmoothCurve(input, a=0.01):
    """
    Using the exponential moving average as described here:
    https://corporatefinanceinstitute.com/resources/capital-markets/exponentially-weighted-moving-average-ewma/
    """
    S_t = torch.zeros(input.size(0))
    S_t[0] = input[0]
    for n, Y_t in enumerate(input[1:], 1):
        S_t[n] = a*Y_t + (1 - a)*S_t[n-1]
    return S_t

def CheckVerDiff(Models):
    """
    Checks which type of file system a list of models uses and throws an error if they're mixed.
    'Models' must be a list of absolute file paths to the model directories
    """
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
    
    if new != 0:
        return "new"
    else:
        return "old"

def LoadModelGraphs(Models):
    filesys = CheckVerDiff(Models) # Check models are from the same 
    print("CheckVerDiff output:", filesys)
    analytic_dim = ChooseAnalytics(filesys) # list of the different analytics to compare
    models_data = GetDataAlg(Models, analytic_dim, gen=filesys) #Returns the data in (model,dim) hierarchy
    DisplayGraphs(analytic_dim, models_data, Models, filesys)

models_loc = "Trained_Models"
Inference_dir = "Inference_Run"
models = os.listdir(models_loc)

ListPrint(models)
modelchoices = []
while True:
    choice  = input("please input modelnum (blank space to stop): ")
    if choice == "":
        break
    else:
        modelchoices.append(models_loc + "/"  + models[int(choice)])

LoadModelGraphs(modelchoices)