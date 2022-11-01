#Custom GANs model goes here 
from email.mime import image
from math import expm1
from turtle import forward
from xdrlib import ConversionError
import torch
from torch import nn


"""
Dev notes:
    - Implement the Generator class
    - Implement the Discriminator class
    
    - Implement the L1 loss function
    
    - Implement a training loop
    
    - After the first successfull training iteration, add a local loss.

"""
#sizes and functions based on the pix2pix paper and github codebase

# Unet structure for the GAN
class UnetEncoderLayer(nn.Module):
    def __init__(self, channel_in, channel_out, normalize=True, dropout=0.0):
        super(UnetEncoderLayer, self).__init__()
        layers = [nn.Conv2d(channel_in, channel_out, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(channel_out))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)       
        
    def forward(self, input):
        return self.model(input)

class UnetDecoderLayer(nn.Module):
    def __init__(self, channel_in, channel_out, dropout=0.0):
        super(UnetDecoderLayer, self).__init__()
        layers = [nn.ConvTranspose2d(channel_in, channel_out, 4, 2, 1, bias=False),
                  nn.InstanceNorm2d(channel_out),
                  nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
            
        self.model = nn.Sequential(*layers)       
        
    def forward(self, input, skip_input):
        input = self.model(input)
        return torch.cat((input, skip_input), 1)


#Discriminator Decode block

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super(ConvRelu, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_, out, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, input):
        return self.model(input)
    

#Pix2pix Unet type, with optimized values from the papers github page, with help from Hemin.
class Generator_Unet1(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(Generator_Unet1, self).__init__()
        #Encoder structure
        self.encode_layer_1 = UnetEncoderLayer(input_channels, 64, normalize=False)
        self.encode_layer_2 = UnetEncoderLayer(64, 128)
        self.encode_layer_3 = UnetEncoderLayer(128, 256)
        self.encode_layer_4 = UnetEncoderLayer(256, 512, dropout=0.5)
        self.encode_layer_5 = UnetEncoderLayer(512, 512, dropout=0.5)
        self.encode_layer_6 = UnetEncoderLayer(512, 512, dropout=0.5)
        self.encode_layer_7 = UnetEncoderLayer(512, 512, dropout=0.5)        
        self.encode_layer_8 = UnetEncoderLayer(256, 512, normalize=False, dropout=0.5)
        
        #Decoder structure
        self.decode_layer_1 = UnetDecoderLayer(512, 512, dropout=0.5)
        self.decode_layer_2 = UnetDecoderLayer(1024, 512, dropout=0.5)
        self.decode_layer_3 = UnetDecoderLayer(1024, 512, dropout=0.5)
        self.decode_layer_4 = UnetDecoderLayer(1024, 512, dropout=0.5)
        self.decode_layer_5 = UnetDecoderLayer(1024, 256)
        self.decode_layer_6 = UnetDecoderLayer(512, 128)
        self.decode_layer_7 = UnetDecoderLayer(256, 64)

        self.final_decoder_layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0 , 1, 0)),
            nn.Conv2d(128, output_channels, 4, padding=1),
            nn.Tanh(),
        )


    def forward(self, input):
        #Encode steps
        e1 = self.encode_layer_1(input)
        e2 = self.encode_layer_2(e1)
        e3 = self.encode_layer_3(e2)
        e4 = self.encode_layer_4(e3)
        e5 = self.encode_layer_5(e4)
        e6 = self.encode_layer_6(e5)
        e7 = self.encode_layer_7(e7)
        e8 = self.encode_layer_8(e7)
        
        #Decode steps
        d1 = self.decode_layer_1(e8, e7)
        d2 = self.decode_layer_2(d1, e6)
        d3 = self.decode_layer_3(d2, e5)
        d4 = self.decode_layer_4(d3, e4)
        d5 = self.decode_layer_5(d4, e3)
        d6 = self.decode_layer_6(d5, e2)
        d7 = self.decode_layer_7(d6, e1)
        
        return self.final_decoder_layer(d7)


# Discriminator function block
def Discriminator_block(input_filters, output_filters, normalization=True):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(input_filters, output_filters, 4, stride=2, padding=1)]
    if normalization:
        layers.append(nn.InstanceNorm2d(output_filters))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


#Discriminator class, Fully convolutional PatchGan discriminator

class Discriminator_1(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator_1, self).__init__()
        
        self.model = nn.Sequential(
            *Discriminator_block(input_channels * 2, 64, normalization=False),
            *Discriminator_block(64, 128),
            *Discriminator_block(128, 256),
            *Discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False),
        )
        
    def forward(self, image_A, image_B):
        #Concatenates original and generates image by channel to produce discriminator input.
        input = torch.cat((image_A, image_B), 1)
        return self.model(input)
    
