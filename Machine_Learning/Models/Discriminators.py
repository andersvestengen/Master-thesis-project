import torch
from torch import nn
import functools


class AttentionLayer(nn.Module):
    def __init__(self, channel_in=3):
        super(AttentionLayer, self).__init__()

        # 1x1 Conv layers
        self.k_layer = nn.Conv2d(channel_in, channel_in // 8,kernel_size=1)
        self.q_layer = nn.Conv2d(channel_in, channel_in // 8,kernel_size=1)
        self.v_layer = nn.Conv2d(channel_in, channel_in,kernel_size=1)
    
        # Initializing the learnable attention variable
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        #Input size: (B,C,H,W)
        B, C, H, W = input.size()
        query_layer = self.q_layer(input).view(B, -1, W*H).permute(0,2,1) #Changes input from 4 tensor to 3 tensor, combining W*H, then permutes input so size is now [B,N,C]
        key_layer = self.k_layer(input).view(B,-1, W*H) # simply truncates 4-dim input tensor to [B,C,N]
        value_layer = self.v_layer(input).view(B,-1,W*H) # simply truncates 4-dim input tensor to [B,C,N]

        attn_energy = torch.bmm(query_layer, key_layer) #BMM [B,N,C] . [B,C,N] -> [B,N,N]
        attn_map = self.softmax(attn_energy)
        attn_weight = torch.bmm(value_layer, attn_map.permute(0,2,1)).view(B,C,W,H) # Original code does attnmap.view(0,2,1) (which would switch tensor shape from BNN to BNN??), but there's no difference to the output so I'm not including it
        #attn_weight = torch.bmm(value_layer, attn_map).view(B,C,W,H) # Original code does attnmap.view(0,2,1) (which would switch tensor shape from BNN to BNN??), but there's no difference to the output so I'm not including it
        output = torch.add(self.gamma * attn_weight, input)

        return output
    
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
    
class Discriminator_layer(nn.Module):
    def __init__(self, channel_in, channel_out, last=False, snormalization=True, batchnorm=False, dropout=False):
        super(Discriminator_layer, self).__init__()
        if snormalization:
            layers = [nn.utils.parametrizations.spectral_norm(nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=1, padding=0))
                    ]
        else:
            layers = [nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=1, padding=0)
                    ]
            
        if dropout:
            layers.append(nn.Dropout(0.5))

        if batchnorm:
            layers.append(nn.BatchNorm2d(channel_out))

        if not last:    
            layers.append(nn.LeakyReLU(0.2, True))            

        self.model = nn.Sequential(*layers)       

    def forward(self, input):
        output =  self.model(input)
        return output

class PixelDiscriminator(nn.Module):

    def __init__(self, input_channels=3, snormalization=True, batchnorm=False, dropout=False):


        super(PixelDiscriminator, self).__init__()

        self.name = "PixelDiscriminator"
        """
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        """
        model = [
            Discriminator_layer(input_channels*2, 32, snormalization=snormalization, batchnorm=batchnorm, dropout=dropout),
            Discriminator_layer(32, 64, snormalization=snormalization, batchnorm=batchnorm, dropout=dropout),
            Discriminator_layer(64, 128, snormalization=snormalization, batchnorm=batchnorm, dropout=dropout),
            Discriminator_layer(128, 256, snormalization=snormalization, batchnorm=batchnorm, dropout=dropout),
            Discriminator_layer(256, 512, snormalization=snormalization, batchnorm=batchnorm, dropout=dropout),
            Discriminator_layer(512, 1, last=True, snormalization=snormalization, batchnorm=batchnorm, dropout=dropout),
                ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        output = self.model(input)
        return output.squeeze()


class Spectral_Conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Spectral_Conv_layer, self).__init__()

        self.layer = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)),
            nn.LeakyReLU(0.1),
        )

    def forward(self, input):
        return self.layer(input)

class SpectralDiscriminator(nn.Module):

    def __init__(self, input_channels=3):


        super(SpectralDiscriminator, self).__init__()

        self.name = "SpectralPixelDiscriminator"
        self.layer1 = Spectral_Conv_layer(input_channels*2, 64)
        self.layer2 = Spectral_Conv_layer(64, 128)
        self.layer3 = Spectral_Conv_layer(128, 256)
        self.layer4 = Spectral_Conv_layer(256, 512)
        self.layer6 = Spectral_Conv_layer(512, 1)

        self.attn1 = AttentionLayer(256)
        self.attn2 = AttentionLayer(512)
        

    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.attn1(output)
        output = self.layer4(output)
        output = self.attn2(output)
        output = self.layer6(output)
        return output.squeeze()
        