import torch.nn as nn
import torch
import torch.nn.functional as F

import importlib.util
import os

current_dir = os.path.dirname(os.path.abspath(__file__))   


import small_sub_module as ssm

class FastSwimmerCNN(nn.Module):
    def __init__(self, params):
        super(FastSwimmerCNN, self).__init__()

        # Parameters for the Descending Arm
        self.encode1 = ssm.CompetitiveEncoderBlockInput(params)
        params['num_channels'] = params['num_filters']
        self.encode2 = ssm.CompetitiveEncoderBlock(params)
        self.encode3 = ssm.CompetitiveEncoderBlock(params)
        self.encode4 = ssm.CompetitiveEncoderBlock(params)
        self.bottleneck = ssm.CompetitiveDenseBlock(params)

        # Parameters for the Ascending Arm
        params['num_channels'] = params['num_filters']
        self.decode4 = ssm.CompetitiveDecoderBlock(params)
        self.decode3 = ssm.CompetitiveDecoderBlock(params)
        self.decode2 = ssm.CompetitiveDecoderBlock(params)
        self.decode1 = ssm.CompetitiveDecoderBlock(params)

        params['num_channels'] = params['num_filters']
        self.classifier = ssm.ClassifierBlock(params)

        # Code for Network Initialization

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def features(self, x):
        encoder_output1, skip_encoder_1, indices_1 = self.encode1.forward(x)
        encoder_output2, skip_encoder_2, indices_2 = self.encode2.forward(encoder_output1)
        encoder_output3, skip_encoder_3, indices_3 = self.encode3.forward(encoder_output2)
        encoder_output4, skip_encoder_4, indices_4 = self.encode4.forward(encoder_output3)

        bottleneck = self.bottleneck(encoder_output4)

        decoder_output4 = self.decode4.forward(bottleneck, skip_encoder_4, indices_4)
        decoder_output3 = self.decode3.forward(decoder_output4, skip_encoder_3, indices_3)
        decoder_output2 = self.decode2.forward(decoder_output3, skip_encoder_2, indices_2)
        decoder_output1 = self.decode1.forward(decoder_output2, skip_encoder_1, indices_1)
        
        return decoder_output1
    
    
    def forward(self, x, grad_cam = False) -> torch.Tensor:
        """
        Computational graph
        :param torch.Tensor x: input image
        :return torch.Tensor logits: prediction logits
        """
        x = self.features(x)

        # Grad-CAM hook
        if grad_cam:
            x.register_hook(self.activation_hook)
        logits = self.classifier.forward(x)

        return logits

    def activation_hook(self, grad):
        self.gradients = grad

    def grad_cam_batch(self, x: torch.Tensor, seg_class: int) -> torch.Tensor:
        distribution = self.forward(x, grad_cam=True)

        # Obtener el gradiente de la clase "seg_class" con respecto a la salida
        self.zero_grad()
        distribution[:, seg_class].sum(dim=1).backward(torch.ones_like(distribution[:, seg_class]).sum(dim=1))

        # Obtener el gradiente de la capa convolucional con respecto a la salida
        grads_val = self.get_activation_gradient()

        # Promedio de gradientes ultima capa convolucional, por canal (alpha_k en el paper)
        weights = grads_val.mean(dim=(2, 3)).squeeze()

        # Obtener mapas de activación de la última capa convolucional (A_k en el paper)
        activations = self.get_activation(x).detach()

        # Promediar los mapas de activacion por canal (sum alpha_k * A_k en el paper)
        for i in range(weights.shape[0]):
            activations[:, i, :, :] *= weights[i]

        # Sumar los mapas de activacion ponderados por los pesos (positivos para la clase)
        heatmap = F.relu(torch.mean(activations, dim=1).squeeze())

        # Normalizar el mapa de calor
        heatmap /= torch.max(heatmap)
        return heatmap, torch.argmax(distribution, dim=1)

    # extract gradient
    def get_activation_gradient(self):
        return self.gradients

    # extract the activation after the last ReLU
    def get_activation(self, x):
        return self.features(x)

    def grad_cam(self, x: torch.Tensor, seg_class: int) -> torch.Tensor:
        distribution = self.forward(x, grad_cam=True)

        # Obtener el gradiente de la clase "seg_class" con respecto a la salida
        self.zero_grad()
        distribution[0, seg_class, :, :].sum().backward()

        # Obtener el gradiente de la capa convolucional con respecto a la salida
        grads_val = self.get_activation_gradient()

        # Promedio de gradientes ultima capa convolucional, por canal (alpha_k en el paper)
        weights = grads_val.mean(dim=(2, 3)).squeeze()

        # Obtener mapas de activación de la última capa convolucional (A_k en el paper)
        activations = self.get_activation(x).detach()

        # Promediar los mapas de activacion por canal (sum alpha_k * A_k en el paper)
        for i in range(weights.shape[0]):
            activations[:, i, :, :] *= weights[i]

        # Sumar los mapas de activacion ponderados por los pesos (positivos para la clase)
        heatmap = F.relu(torch.mean(activations, dim=1).squeeze())

        # Normalizar el mapa de calor
        heatmap /= torch.max(heatmap)
        return heatmap, torch.argmax(distribution, dim=1)
    
def test():
    # Parameters for the Model
    params = {
        "num_channels": 7,
        "num_filters": 64,
        "kernel_h": 5,
        "kernel_w": 5,
        "stride_conv": 1,
        "pool": 2,
        "stride_pool": 2,
        "num_classes": 44,
        "kernel_c": 1,
        "input": True,
    }

    # Define the Model
    model = FastSwimmerCNN(params)

    # Define the Input
    x = torch.rand(1, 7, 256, 256)

    # Forward Pass
    out = model.forward(x)

    # Print the Output Shape
    print(out.shape)

    return out

if __name__ == "__main__":
    test()