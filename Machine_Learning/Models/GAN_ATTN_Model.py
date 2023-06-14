import torch
from torch import nn
import functools
from torchvision.models import resnet34, ResNet34_Weights

"""
Based on:
https://towardsdatascience.com/building-your-own-self-attention-gans-e8c9b9fe8e51
https://jonathan-hui.medium.com/gan-self-attention-generative-adversarial-networks-sagan-923fccde790c
https://arxiv.org/pdf/1805.08318.pdf
"""

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
    

class UnetEncoderLayer(nn.Module):
    def __init__(self, channel_in, channel_out, normalize=False, dropout=0.0):
        super(UnetEncoderLayer, self).__init__()
        layers = [nn.utils.parametrizations.spectral_norm(nn.Conv2d(channel_in, channel_out, 4, 2, 1, bias=True))]
        if normalize:
            layers.append(nn.LeakyReLU(0.1))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)       
        
        
    def forward(self, input):
        output = self.model(input)
        return output

class UnetDecoderLayer(nn.Module):
    def __init__(self, channel_in, channel_out, dropout=0.0, normalize=False, attention=False):
        super(UnetDecoderLayer, self).__init__()
        if normalize:
            layers = [nn.utils.parametrizations.spectral_norm(nn.ConvTranspose2d(channel_in, channel_out, 4, 2, 1, bias=True)),
                    nn.LeakyReLU(0.1),
            ]
        else:
            layers = [nn.utils.parametrizations.spectral_norm(nn.ConvTranspose2d(channel_in, channel_out, 4, 2, 1, bias=True)),
                    nn.LeakyReLU(0.1),
            ]
        if dropout:
            layers.append(nn.Dropout(dropout))
            
        if attention:
            layers.append(AttentionLayer(channel_out))
        self.model = nn.Sequential(*layers)       
        
        

    def forward(self, input, skip_input):
        modelinput = torch.cat((input, skip_input), 1)
        modelout = self.model(modelinput)
        return modelout
    

#Modified Pix2Pix Unet now with attention layers,skip connections and pretrained Encoder-side.
#Closer to a SAGAN implementation
class Generator_Unet_Attention(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(Generator_Unet_Attention, self).__init__()
        #Encoder structure
        self.name = "Generator_Unet_Attention"

        dilation_layer = [nn.Conv2d(512, 512, 1, dilation=1),
                        nn.ELU(inplace=True),
                        ]

        self.conv1 = UnetEncoderLayer(input_channels, 64)
        self.conv2 = UnetEncoderLayer(64, 64)
        self.conv3 = UnetEncoderLayer(64, 128, dropout=0.25)
        self.conv4 = UnetEncoderLayer(128, 256, dropout=0.25)
        self.conv5 = UnetEncoderLayer(256, 512, dropout=0.25)
        

        #Decoder structure
        self.decode_layer_1 = UnetDecoderLayer(1024, 256, dropout=0.25) # 512 + 512
        self.decode_layer_2 = UnetDecoderLayer(512,  128, dropout=0.25) # 256 + 256
        self.decode_layer_3 = UnetDecoderLayer(256, 64, dropout=0.25, attention=True) # 128 + 128
        self.decode_layer_4 = UnetDecoderLayer(128, 64, attention=True) # 64 + 64


        self.final_decoder_layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0 , 1, 0)),
            nn.Conv2d(64, output_channels, 4, padding=1),
            nn.Tanh(),
        )


    def forward(self, input):
        #Encoder
        e1 = self.conv1(input)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)

        
        #Center
        #Can look to add more dilated convolutions in the future
        #Decoder
        d1 = self.decode_layer_1(e5, e4)
        d2 = self.decode_layer_2(d1, e3)
        d3 = self.decode_layer_3(d2, e2)
        d4 = self.decode_layer_4(d3, e1)
        
        return self.final_decoder_layer(d4)



class Generator_Unet_Window_Attention(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(Generator_Unet_Attention, self).__init__()
        #Encoder structure
        self.name = "Generator_Unet_Attention"

        dilation_layer = [nn.Conv2d(256, 256, 1, dilation=1),
                        nn.ELU(inplace=True),
                        ]

        self.conv1 = UnetEncoderLayer(input_channels, 64)
        self.conv2 = UnetEncoderLayer(64, 64)
        self.conv3 = UnetEncoderLayer(64, 128, dropout=0.25)
        self.conv4 = UnetEncoderLayer(128, 256, dropout=0.25)
        
        self.dilation_1 = nn.Sequential(*dilation_layer)
        
        self.dilation_2 = nn.Sequential(*dilation_layer)
    
        self.dilation_3 = nn.Sequential(*dilation_layer)
    
        self.dilation_4 = nn.Sequential(*dilation_layer)

        #Decoder structure
        self.decode_layer_1 = UnetDecoderLayer(512,  128, dropout=0.25) # 256 + 256
        self.decode_layer_2 = UnetDecoderLayer(256, 64, dropout=0.25) # 128 + 128
        self.decode_layer_3 = UnetDecoderLayer(128, 64) # 64 + 64

        self.final_decoder_layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0 , 1, 0)),
            nn.Conv2d(64, output_channels, 4, padding=1),
            nn.Tanh(),
        )


    def forward(self, input):
        #Encoder
        e1 = self.conv1(input)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        
        #Center
        #Can look to add more dilated convolutions in the future
        c1 = self.dilation_1(e4)
        c2 = self.dilation_2(c1)
        c3 = self.dilation_3(c2)
        c4 = self.dilation_4(c3)

        #Decoder
        d1 = self.decode_layer_1(c4, e4)
        d2 = self.decode_layer_2(d1, e3)
        d3 = self.decode_layer_3(d2, e2)
        
        return self.final_decoder_layer(d3)