from typing import Any
import torch
from torch import nn

class LossFunctions(nn.Module):

    def __init__(self, device, Discriminator, Settings):
        super(LossFunctions, self).__init__()
        self.device = device
        self.Settings = Settings
        self.Discriminator = Discriminator
        if Settings["Loss"] == "CGAN":
            self.CGAN_loss = nn.BCEWithLogitsLoss().to(self.device)
        self.pixelwise_loss = nn.L1Loss().to(self.device)
        self.pixelwise_local_loss = nn.MSELoss().to(self.device)
        self.Latent_Feature_Criterion = nn.MSELoss().to(self.device)
        self.lambda_gp = Settings["lambda_gp"]

    # Helper functions -----------------------------------------

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
            label_tensor = torch.tensor(0, device=self.device, dtype=torch.float32, requires_grad=False)

        return label_tensor.expand_as(tensor_size)
    


    # Loss functions -------------------------------------------
    # Inputs are always fake, pred
    # loss function inputs must always be in the order ( *input, *target ) !
    def Generator_Autoencoder_Pixellos(self, input, target):
        return self.pixelwise_loss(input, target)

    def Generator_Coordinate_Pixelloss(self, real_B, fake_B, defect_coordinates): 
        SampleY, SampleX, BoxSize = defect_coordinates[0]

        loss_pixel = self.pixelwise_loss(fake_B, real_B)
        
        L1_loss_region = (BoxSize * int(self.Settings["Loss_region_Box_mult"])).to(self.device)
        SampleY, SampleX = self.CenteringAlgorithm(BoxSize, L1_loss_region, SampleY, SampleX)
        local_pixelloss = self.pixelwise_loss(fake_B[:,:,SampleY:SampleY+L1_loss_region,SampleX:SampleX+L1_loss_region], real_B[:,:,SampleY:SampleY+L1_loss_region,SampleX:SampleX+L1_loss_region])
        
        return loss_pixel, local_pixelloss

    def Generator_Pixelloss(self, fake_B, real_B, mask):

        local_pixelloss = self.pixelwise_loss(torch.where(mask, 0, fake_B), torch.where(mask, 0, real_B)) # defect-region
        General_pixelloss = self.pixelwise_local_loss(torch.where(~mask, 0, fake_B), torch.where(~mask, 0, real_B)) # everywhere else

        return General_pixelloss, local_pixelloss

    def Generator_Autoencoder_Pixelloss(self, fake_B, real_B):

        pixelloss = self.pixelwise_loss(fake_B, real_B)

        return pixelloss

    def LatentFeatureLoss(self, input, target):

        return self.Latent_Feature_Criterion(input, target)

    def Latent_WGAN_Loss(self, input, target):
        loss_real = - torch.mean(target)
        loss_fake = torch.mean(input)

        return loss_real + loss_fake
    
    def WGAN_Discriminator(self, *args):
        fake_pred, real_pred = args

        loss_real = - torch.mean(real_pred)
        loss_fake = torch.mean(fake_pred)

        return loss_real + loss_fake

    def WGANGP_Discriminator(self, *args):
        fake_pred, real_pred, fake_BA, real_AB = args
        loss_real = - torch.mean(real_pred)
        loss_fake = torch.mean(fake_pred)

        gp_term = self.Gradient_Penalty(real_AB, fake_BA)
        return loss_real + loss_fake + gp_term


    def WGAN_Generator(self, fake_pred):
        return - torch.mean(fake_pred)

    def CGAN_Dual_Encoder_Discriminator(self, *args): 
        fake_in, real_in = args
        real = self.Make_Label_Tensor(real_in, 1)
        fake = self.Make_Label_Tensor(fake_in, 0)

        return ( self.CGAN_loss(real_in, real) + self.CGAN_loss(fake_in, fake) ) * 0.5

    def CGAN_Discriminator(self, *args): 
        fake_pred, real_pred = args
        real = self.Make_Label_Tensor(real_pred, 1)
        fake = self.Make_Label_Tensor(fake_pred, 0)

        return self.CGAN_loss(real_pred, real) + self.CGAN_loss(fake_pred, fake) * 0.5 

    def CGAN_Generator(self, fake_pred): # Losses must be supplied as: ( *input, *target ) !
        real = self.Make_Label_Tensor(fake_pred, 1)
        return self.CGAN_loss(fake_pred, real)

    def Gradient_Penalty(self, real_AB, fake_AB):
        #Create interpolation term
        alpha = torch.rand((self.Settings["batch_size"], 1, 1, 1), device=self.device)

        #Create interpolates
        interpolates = (alpha * real_AB + ((1 - alpha)* fake_AB)).requires_grad_(True)
        Discriminator_interpolates = self.Discriminator(interpolates)
        fake = torch.ones(Discriminator_interpolates.size(), device=self.device).requires_grad_(True)
        #get gradients w.r.t interpolates

        gradients = torch.autograd.grad(
            outputs=Discriminator_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        #Flatten array
        gradients = gradients[0].view(real_AB.size(0), -1)
        # Calculate and return GP
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) -1) ** 2).mean() * self.lambda_gp
        return gradient_penalty
    
    def Hinge_loss_Discriminator(self, *args):
        fake_pred, real_pred = args
        return torch.relu(torch.mean(1 - real_pred)) + torch.relu(torch.mean(1 + fake_pred))

    def Hinge_loss_Generator(self, predicted_fake):
        return -torch.mean(predicted_fake)