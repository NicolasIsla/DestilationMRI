# En main.py se genera un diccionario "args" y se llama a fastsurfercnn. En eval.py dentro de fastsurfercnn() se llama a la funcion FastSurferCNN(), la cual recibe un diccionario "params" (que es args) como argumento. En networks.py esta esa funcion FastSurferCNN(), la cual inicializa los bloques densos competitivos aqui descritos con el diccionario "params", y luego da forma a la red completa. En FastSurferCNN, luego de generar el bloque denso de entrada, el parametro "num channels" del diccionario (que son los canales de entrada, por defecto 7) se setea como el mismo que "num filters", ya que los bloques que siguen no son de entrada y solo necesitan saber la cantidad de neuronas de las capas anteriores, que por defecto son 64

# IMPORTS
import torch
import torch.nn as nn


# Building Blocks
class CompetitiveDenseBlock(nn.Module):
    """
    Function to define a competitive dense block comprising of 3 convolutional layers, with BN/ReLU

    Inputs:
    -- Params
     params = {'num_channels': 1,
               'num_filters': 64,
               'kernel_h': 5,
               'kernel_w': 5,
               'stride_conv': 1,
               'pool': 2,
               'stride_pool': 2,
               'num_classes': 44
               'kernel_c':1
               'input':True
               }
    """

    def __init__(self, params, outblock=False):
        super(CompetitiveDenseBlock, self).__init__()

        # Padding to get output tensor of same dimensions
        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        # Sub-layer output sizes for BN; and
        conv0_in_size = int(params['num_filters'])  # num_channels
        conv0_out_size = int(params['num_filters'])

        # Define the learnable layers

        self.conv0 = nn.Conv2d(in_channels=conv0_in_size, out_channels=conv0_out_size,
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               stride=params['stride_conv'], padding=(padding_h, padding_w))

        self.bn0 = nn.BatchNorm2d(num_features=conv0_out_size)

        self.prelu = nn.PReLU()  # Learnable ReLU Parameter

    def forward(self, x):
        # Activation from pooled input
        x0_act = self.prelu(x)

        # Convolution block 1
        x0 = self.conv0(x0_act)
        out = self.bn0(x0)

        return out


class CompetitiveDenseBlockInput(nn.Module):
    """
    Function to define a competitive dense block comprising of 3 convolutional layers, with BN/ReLU for input

    Inputs:
    -- Params
     params = {'num_channels': 1,
               'num_filters': 64,
               'kernel_h': 5,
               'kernel_w': 5,
               'stride_conv': 1,
               'pool': 2,
               'stride_pool': 2,
               'num_classes': 44
               'kernel_c':1
               'input':True
              }
    """

    def __init__(self, params):
        super(CompetitiveDenseBlockInput, self).__init__()

        # Padding to get output tensor of same dimensions
        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        # Sub-layer output sizes for BN; and
        conv0_in_size = int(params['num_channels'])
        conv0_out_size = int(params['num_filters'])

        # Define the learnable layers

        self.conv0 = nn.Conv2d(in_channels=conv0_in_size, out_channels=conv0_out_size,
                               kernel_size=(params['kernel_h'], params['kernel_w']),
                               stride=params['stride_conv'], padding=(padding_h, padding_w))

        self.bn0 = nn.BatchNorm2d(num_features=conv0_in_size)
        self.bnout = nn.BatchNorm2d(num_features=conv0_out_size)


    def forward(self, x):
        # Input batch normalization
        x0_bn = self.bn0(x)

        # Convolution block1
        x0 = self.conv0(x0_bn)
        out = self.bnout(x0)

        return out


class CompetitiveEncoderBlock(CompetitiveDenseBlock):
    def __init__(self, params):
        super(CompetitiveEncoderBlock, self).__init__(params)
        self.maxpool = nn.MaxPool2d(kernel_size=params['pool'], stride=params['stride_pool'],
                                    return_indices=True)  # For Unpooling later on with the indices

    def forward(self, x):
        out_block = super(CompetitiveEncoderBlock, self).forward(x)  # To be concatenated as Skip Connection
        out_encoder, indices = self.maxpool(out_block)  # Max Pool as Input to Next Layer
        return out_encoder, out_block, indices



class CompetitiveEncoderBlockInput(CompetitiveDenseBlockInput):
    def __init__(self, params):
        super(CompetitiveEncoderBlockInput, self).__init__(params)  # The init of CompetitiveDenseBlock takes in params
        self.maxpool = nn.MaxPool2d(kernel_size=params['pool'], stride=params['stride_pool'],
                                    return_indices=True)  # For Unpooling later on with the indices

    def forward(self, x):
        out_block = super(CompetitiveEncoderBlockInput, self).forward(x)  # To be concatenated as Skip Connection
        out_encoder, indices = self.maxpool(out_block)  # Max Pool as Input to Next Layer
        return out_encoder, out_block, indices


class CompetitiveDecoderBlock(CompetitiveDenseBlock):
    def __init__(self, params, outblock=False):
        super(CompetitiveDecoderBlock, self).__init__(params, outblock=outblock)
        self.unpool = nn.MaxUnpool2d(kernel_size=params['pool'], stride=params['stride_pool'])

    def forward(self, x, out_block, indices):
        unpool = self.unpool(x, indices)
        unpool = torch.unsqueeze(unpool, 4)

        out_block = torch.unsqueeze(out_block, 4)
        concat = torch.cat((unpool, out_block), dim=4)  # Competitive Concatenation
        concat_max, _ = torch.max(concat, 4)
        out_block = super(CompetitiveDecoderBlock, self).forward(concat_max)

        return out_block


class ClassifierBlock(nn.Module):
    def __init__(self, params):
        super(ClassifierBlock, self).__init__()
        self.conv = nn.Conv2d(params['num_channels'], params['num_classes'], params['kernel_c'],
                              params['stride_conv'])  # To generate logits

    def forward(self, x):
        logits = self.conv(x)

        return logits
