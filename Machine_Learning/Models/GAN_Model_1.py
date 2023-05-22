#Custom GANs model goes here 
#from email.mime import image
#from math import expm1
#from turtle import forward
#from xdrlib import ConversionError
import torch
from torch import nn
import functools
"""
Dev notes:
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
        output = self.model(input)
        return output

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
        self.name = "Generator_Unet1"
        self.encode_layer_1 = UnetEncoderLayer(input_channels, 64, normalize=False)
        self.encode_layer_2 = UnetEncoderLayer(64, 128)
        self.encode_layer_3 = UnetEncoderLayer(128, 256)
        self.encode_layer_4 = UnetEncoderLayer(256, 512, dropout=0.5)
        self.encode_layer_5 = UnetEncoderLayer(512, 512, dropout=0.5)
        self.encode_layer_6 = UnetEncoderLayer(512, 512, dropout=0.5)
        self.encode_layer_7 = UnetEncoderLayer(512, 512, dropout=0.5)        
        self.encode_layer_8 = UnetEncoderLayer(512, 512, normalize=False, dropout=0.5)
        
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
        e7 = self.encode_layer_7(e6)
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
        self.name = "Discriminator_1"
        self.model = nn.Sequential(
            *Discriminator_block(input_channels * 2, 64, normalization=False),
            *Discriminator_block(64, 128),
            *Discriminator_block(128, 256),
            *Discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False),
        )
        
    def forward(self, input):
        #Concatenates original and generates image by channel to produce discriminator input.
        #input = torch.cat((image_A, image_B), 1)
        return self.model(input)
    

class PixPatchGANDiscriminator(nn.Module):
    
    def __init__(self, input_ch, output_filters=64, num_layers=3, norm_layer=nn.BatchNorm2d):
        """
        input_ch:       number of channels in the input images
        outputfilers:   number of filters in the last output layer
        num_layers:     number of layers in the discriminator
        norm_layer:     which type of normalization layer is used (batch- or instance-norm) 
        """
        
        super(PixPatchGANDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        Layers = [nn.Conv2d(input_ch*2, output_filters, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]

        number_of_filters = 1
        prev_number_of_filters = 1
        for n in range(1, num_layers):
            prev_number_of_filters = number_of_filters
            number_of_filters = min(2**n, 8)

            Layers += [
                nn.Conv2d(output_filters * prev_number_of_filters, output_filters * number_of_filters, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(output_filters * number_of_filters),
                nn.LeakyReLU(0.2, True)
            ]
        
        prev_number_of_filters = number_of_filters
        number_of_filters = min(2**num_layers, 8)

        Layers += [
            nn.Conv2d(output_filters * prev_number_of_filters, output_filters * number_of_filters, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(output_filters * number_of_filters),
            nn.LeakyReLU(0.2, True)
        ]

        Layers += [nn.Conv2d(output_filters * number_of_filters, 1, kernel_size=4, stride=1, padding=1)] # output-prediction layer

    
        self.model = nn.Sequential(*Layers) # build model

    def forward(self, input):
        return self.model(input)
    


class UnetSkipConnectionBlock(nn.Module):
    """
    Defines the Unet sub-layer with skip connections.
    """

    def __init__(self, outer_layers, inner_layers, input_layers=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """
        outer_layer:        number of filters in the outer convolutional_layer
        inner_layer:        number of filters in the inner convolutional layer
        input_layer:        number of channels in the image inputs
        outermost:          is this the outermost layer?
        submodule:          previously defined submodules (skip connection?)
        innermost:          is this the innermost layer?
        norm_layer:         define which type of normalization layer
        use_drouput:        use dropout in this sub-module?
        """

        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_layers is None:
            input_layers = outer_layers

        downconv    = nn.Conv2d(input_layers, inner_layers, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu    = nn.LeakyReLU(0.2, True)
        downnorm    = norm_layer(inner_layers)
        uprelu      = nn.ReLU(True)
        upnorm      = norm_layer(outer_layers)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_layers * 2, outer_layers, kernel_size=4, stride=2, padding=1) #bias not included in the original github code, but that must be an oversight?
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up

        elif innermost:
            upconv = nn.ConvTranspose2d(inner_layers, outer_layers, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        
        else:
            upconv = nn.ConvTranspose2d(inner_layers * 2, outer_layers, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.outermost:
            return self.model(input)
        else:
            return torch.cat([input, self.model(input)], 1)
        

class UnetGenerator(nn.Module):
    """
    Creates the Pix2Pix U-net style Generator 

    pix2pix discription says that it constructs the U-net from the innermost layer to the outermost layer and that this is a recursive process, which.. yeah.
    """

    def __init__(self, input_channels, output_channels, num_downsamples=8, channels_last_conv=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """
        input_channels:         number of channels in the input images
        output_channels:        number of channels in the output images
        num_downsamples:        multiplicative deciding by how much the images gets downsampled as they traverse to the bottom of the U-net. Deciding on 8 layers as our input is 256x256 in size
        channels_last_conv:     number of channels in the output layer
        norm_layers:            defines which function to be used for the normalization layers
        """


        super(UnetGenerator, self).__init__()

        unet_block = UnetSkipConnectionBlock(channels_last_conv * 8, channels_last_conv * 8, input_layers=None, submodule=None, norm_layer=norm_layer, innermost=True) #Defining the innermost layer first?
        for i in range(num_downsamples - 5): #Why would you subtract exactly five, this must mean there's a smallest network, which is 7?
            unet_block = UnetSkipConnectionBlock(channels_last_conv * 8, channels_last_conv * 8, input_layers=None, submodule=unet_block, norm_layer=norm_layer)
        
        unet_block = UnetSkipConnectionBlock(channels_last_conv * 4, channels_last_conv * 8, input_layers=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(channels_last_conv * 2, channels_last_conv * 4, input_layers=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(channels_last_conv, channels_last_conv * 2, input_layers=None, submodule=unet_block, norm_layer=norm_layer)

        self.model = UnetSkipConnectionBlock(output_channels, channels_last_conv, input_layers=input_channels, submodule=unet_block, outermost=True, norm_layer=norm_layer)

    def forward(self, input):
        return self.model(input)



class PixelDiscriminator(nn.Module):

    def __init__(self, input_channels, last_conv_channels=64, norm_layer=nn.BatchNorm2d):


        super(PixelDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.Conv2d(input_channels*2, last_conv_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(last_conv_channels, last_conv_channels * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(last_conv_channels * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(last_conv_channels * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)
        ]

        self.model = nn.Sequential(*model)

    def forward(self. input):
        return self.model(input)
