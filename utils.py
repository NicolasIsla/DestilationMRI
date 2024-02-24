import os
import torch
from fastsurfer.networks import FastSurferCNN

def load_model(model_name):
    if model_name == 'Sagittal':
        params = {'num_channels': 7,
        'num_filters': 64,
        'kernel_h': 5,
        'kernel_w': 5,
        'stride_conv': 1,
        'pool': 2,
        'stride_pool': 2,
        'num_classes': 51,
        'kernel_c':1,
        'input':True
        }
        model = FastSurferCNN(params)
        dir = os.path.join('checkpoints', 'Sagittal_Weights_FastSurferCNN', 'ckpts', 'Epoch_30_training_state.pkl')
        weights = torch.load(dir, map_location=torch.device('cpu'))
        model.load_state_dict(weights['model_state_dict'])
        return model
    elif model_name == 'Axial':
        params = {'num_channels': 7,
        'num_filters': 64,
        'kernel_h': 5,
        'kernel_w': 5,
        'stride_conv': 1,
        'pool': 2,
        'stride_pool': 2,
        'num_classes': 79,
        'kernel_c':1,
        'input':True
        }

        model = FastSurferCNN(
            params=params,
        )

        dir = os.path.join('checkpoints', 'Axial_Weights_FastSurferCNN', 'ckpts', 'Epoch_30_training_state.pkl')
        weights = torch.load(dir,
                            map_location=torch.device('cpu'))

        model.load_state_dict(weights['model_state_dict'])
        return model
    
    elif model_name == 'Coronal':
        params = {'num_channels': 7,
        'num_filters': 64,
        'kernel_h': 5,
        'kernel_w': 5,
        'stride_conv': 1,
        'pool': 2,
        'stride_pool': 2,
        'num_classes': 79,
        'kernel_c':1,
        'input':True
        }
    # checkpoints\Coronal_Weights_FastSurferCNN\ckpts\Epoch_30_training_state.pkl
        dir = os.path.join('checkpoints', 'Coronal_Weights_FastSurferCNN', 'ckpts', 'Epoch_30_training_state.pkl')
        weights = torch.load(dir,
                            map_location=torch.device('cpu'))
        model = FastSurferCNN(params)
        model.load_state_dict(weights['model_state_dict'])
        return model
        
    else:
        raise ValueError("Model not found")