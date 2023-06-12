import torch
from torch import nn
import functools

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
    
    def __init__(self, input_ch=3, output_filters=64, num_layers=3, norm_layer=nn.BatchNorm2d):
        """
        input_ch:       number of channels in the input images
        outputfilers:   number of filters in the last output layer
        num_layers:     number of layers in the discriminator
        norm_layer:     which type of normalization layer is used (batch- or instance-norm) 
        """
        
        super(PixPatchGANDiscriminator, self).__init__()
        self.name = "PixPatchGANDiscriminator"
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

            if norm_layer is not None:
                Layers += [
                    nn.Conv2d(output_filters * prev_number_of_filters, output_filters * number_of_filters, kernel_size=4, stride=2, padding=1, bias=use_bias),
                    norm_layer(output_filters * number_of_filters),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                Layers += [
                    nn.Conv2d(output_filters * prev_number_of_filters, output_filters * number_of_filters, kernel_size=4, stride=2, padding=1, bias=use_bias),
                    nn.LeakyReLU(0.2, True)
                ]
        
        prev_number_of_filters = number_of_filters
        number_of_filters = min(2**num_layers, 8)

        if norm_layer is not None:
            Layers += [
                nn.Conv2d(output_filters * prev_number_of_filters, output_filters * number_of_filters, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(output_filters * number_of_filters),
                nn.LeakyReLU(0.2, True)
            ]
        else:
            Layers += [
                nn.Conv2d(output_filters * prev_number_of_filters, output_filters * number_of_filters, kernel_size=4, stride=2, padding=1, bias=use_bias),
                nn.LeakyReLU(0.2, True)
            ]

        Layers += [nn.Conv2d(output_filters * number_of_filters, 1, kernel_size=4, stride=1, padding=1)] # output-prediction layer

    
        self.model = nn.Sequential(*Layers) # build model

    def forward(self, input):
        return self.model(input)
    


class PixelDiscriminator(nn.Module):

    def __init__(self, input_channels=3, last_conv_channels=64, norm_layer=nn.BatchNorm2d, use_bias=False):


        super(PixelDiscriminator, self).__init__()

        self.name = "PixelDiscriminator"
        """
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        """
        if norm_layer is not None:
            model = [
                nn.Conv2d(input_channels*2, last_conv_channels, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(last_conv_channels, last_conv_channels * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
                norm_layer(last_conv_channels * 2),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(last_conv_channels * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)
            ]
        else:
            model = [
                nn.Conv2d(input_channels*2, last_conv_channels, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(last_conv_channels, last_conv_channels * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(last_conv_channels * 2, last_conv_channels * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(last_conv_channels * 2, last_conv_channels * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(last_conv_channels * 2, last_conv_channels * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(last_conv_channels * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)
            ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return torch.flatten(self.model(input))


class Spectral_Conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Spectral_Conv_layer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv = nn.utils.parametrizations.spectral_norm(self.conv)
        self.act = nn.ELU(True)
        self.dropout = nn.Dropout(0.25)

    def forward(self, input):
        return self.dropout(self.act(self.conv(input)))

class SpectralDiscriminator(nn.Module):

    def __init__(self, input_channels=3):


        super(SpectralDiscriminator, self).__init__()

        self.name = "SpectralPixelDiscriminator"
        self.layer1 = Spectral_Conv_layer(input_channels*2, 64)
        self.layer2 = Spectral_Conv_layer(64, 128)
        self.layer3 = Spectral_Conv_layer(128, 256)
        self.layer4 = Spectral_Conv_layer(256, 256)
        self.layer5 = Spectral_Conv_layer(256, 256)
        self.layer6 = Spectral_Conv_layer(256, 1)
        

    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        output = self.layer6(output)
        return torch.flatten(output)
        