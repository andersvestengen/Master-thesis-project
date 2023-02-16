import torch
from torch import nn

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


class DAM_BLOCK(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(DAM_BLOCK, self).__init__()        
    
        self.conv_1x1 = nn.Conv2d(channels_in, channels_out, 1, 1, 0)
        self.M_act = nn.Sequential([
                    nn.Conv2d(channels_out, channels_out, 1, 1, 0),
                    nn.Conv2d(channels_out, channels_out, 1, 1, 0),
                    nn.Sigmoid(),
                    ])

        self.OutputLayers = nn.Sequential([
                    nn.Conv2d(channels_out, channels_out, 1, 1, 0),
                    nn.Conv2d(channels_out, channels_out, 1, 1, 0),
                    nn.Upsample(scale_factor=2),
                    ])



    def forward(self, S, T):
        input = torch.cat((S,T), 1)
        F_i = self.conv1x1(input)
        M_i = self.M_act(F_i) # Attention map
        F_i_mark = torch.mul(M_i, F_i)
        F_out = torch.add(F_i_mark, F_i)
        Output = self.OutputLayers(F_out)

        if nn.Module.training:
            return Output, M_i #Return output and the M_i for training. 
        else:
            return Output



class DAM_Generator_Unet1(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(DAM_Generator_Unet1, self).__init__()
        #Encoder structure
        self.name = "DAM_GEN_Unet1"
        self.encode_layer_1 = UnetEncoderLayer(input_channels, 64, normalize=False)
        self.encode_layer_2 = UnetEncoderLayer(64, 128)
        self.encode_layer_3 = UnetEncoderLayer(128, 256)
        self.encode_layer_4 = UnetEncoderLayer(256, 512, dropout=0.5) #Center convolutuiobn
        self.encode_layer_5 = UnetEncoderLayer(512, 512, dropout=0.5)
        self.encode_layer_6 = UnetEncoderLayer(512, 512, dropout=0.5)
        self.encode_layer_7 = UnetEncoderLayer(512, 512, dropout=0.5)        
        self.encode_layer_8 = UnetEncoderLayer(512, 512, normalize=False, dropout=0.5)
        
        #Decoder structure
        self.decode_layer_1 = UnetDecoderLayer(512, 512, dropout=0.5)

        self.decode_layer_2 = UnetDecoderLayer(1024, 512, dropout=0.5)
        
        self.decode_layer_3 = UnetDecoderLayer(1024, 512, dropout=0.5)
        
        self.decode_layer_4 = UnetDecoderLayer(1024, 512, dropout=0.5)
        self.DAM_layer_1    = DAM_BLOCK(1024, 512)

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