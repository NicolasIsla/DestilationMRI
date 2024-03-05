import os
import torch
from fastsurfer.networks import FastSurferCNN
from fastswimmer.small_networks import FastSwimmerCNN
# import fastswimmer.small_networks as ssm # solucionar el problema de importar small_sub_module
import argparse

def load_model_teacher(model_name):
    if model_name == 'sagittal':
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
        model.num_classes = 51
        dir = os.path.join('checkpoints', 'Sagittal_Weights_FastSurferCNN', 'ckpts', 'Epoch_30_training_state.pkl')
        weights = torch.load(dir, map_location=torch.device('cpu'))
        model.load_state_dict(weights['model_state_dict'])
        return model
    elif model_name == 'axial':
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
        model.num_classes = 79

        dir = os.path.join('checkpoints', 'Axial_Weights_FastSurferCNN', 'ckpts', 'Epoch_30_training_state.pkl')
        weights = torch.load(dir,
                            map_location=torch.device('cpu'))

        model.load_state_dict(weights['model_state_dict'])
        return model
    
    elif model_name == 'coronal':
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
        model.num_classes = 79
        model.load_state_dict(weights['model_state_dict'])
        return model
        
    else:
        raise ValueError("Model not found")
    


def load_model_student(model_name, baseline=False):
    if model_name == 'sagittal':
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
        model = FastSwimmerCNN(params)
        model.num_classes = 51
        if baseline:
            dir = os.path.join('checkpoints', 'Sagittal_Weights_FastSwimmerCNN', 'ckpts', f'Epoch_14_training_state.pt')
            weights = torch.load(dir,
                                map_location=torch.device('cpu'))
            model.load_state_dict(weights)
        return model
    elif model_name == 'axial':
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

        model = FastSwimmerCNN(
            params=params,
        )
        model.num_classes = 79
        if baseline:
            dir = os.path.join('checkpoints', 'Axial_Weights_FastSwimmerCNN', 'ckpts', f'Epoch_14_training_state.pt')
            weights = torch.load(dir,
                                map_location=torch.device('cpu'))
            model.load_state_dict(weights)
        return model

    
    elif model_name == 'coronal':
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
    
        model = FastSwimmerCNN(params)
        model.num_classes = 79
        if baseline:
           dir = os.path.join('checkpoints', 'Coronal_Weights_FastSwimmerCNN', 'ckpts', f'Epoch_14_training_state.pt')
           weights = torch.load(dir,
                               map_location=torch.device('cpu'))
           model.load_state_dict(weights)
        return model
        
    else:
        raise ValueError("Model not found")
    
import os
import argparse

class DotDict(dict):
    def __getattr__(self, item):
        if item in self:
            return self[item]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_common_arguments(description='Common arguments'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--dataset', type=str, choices=['sagittal', 'axial', 'coronal'], default='sagittal', help='Dataset to use')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples to use')
    parser.add_argument('--forced', type=int, default=0, help='Force preprocessing')
    parser.add_argument('--data_dir', type=str, default='./data/', help='Data directory')
    parser.add_argument('--dummy', type=int, default=0, help='Use dummy data')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--show_versions', type=int, default=1, help='Show available versions to load from')
        
    parser.add_argument('--device', type=int, default=0, help='Device to use for training')
    parser.add_argument('--version', type=int, default=None, help='Select a version to load from')
    return parser
def get_arguments_trainer():
    parser = get_common_arguments(description='Trainer arguments')
    parser.add_argument('--epochs', type=int, default=5, help='Maximum number of epochs')
    args = parser.parse_args()
    return DotDict(args.__dict__)

def get_arguments_metrics():
    parser = get_common_arguments(description='Metrics arguments')
    # Hacer obligatoria la version si no se dice --show_versions
    if '--show_versions' not in parser._option_string_actions:
        parser._option_string_actions['--version'].required = True
    args = parser.parse_args()
    return DotDict(args.__dict__)

def get_arguments_distiller():
    parser = get_common_arguments(description='Distiller arguments')
    parser.add_argument('--epochs', type=int, default=5, help='Maximum number of epochs')
    parser.add_argument('--distillation_temperature', type=float, default=3.0, help='Distillation temperature')
    parser.add_argument('--alpha', type=float, default=0.5, help='Distillation loss weight')
    args = parser.parse_args()
    return DotDict(args.__dict__)

def get_arguments(log_dir, type):
    import sys
    
    args = getattr(sys.modules[__name__], f"get_arguments_{type}")()
    
    architecture = args["dataset"]
    
    versions = get_versions(log_dir, architecture, args['dataset']) # [0, 1, 2, ...], [] si no hay versiones
        
    # Mostrar las versiones disponibles
    if args['show_versions']:
        print(f"Versions: {versions}")
        exit(0)
    

    # Obtener el directorio del experimento y el checkpoint
    name, exp_dir, ckpt = get_experiment(log_dir, architecture, args['dataset'], args['version'])

    # Cargar el datamodule
    dm = get_data_module(args['dataset'], args['batch_size'], f"{args['data_dir']}/{args['dataset']}/", args['samples'], args['forced'], args['dummy'], args["device"])
    dm.prepare_data()
    dm.setup()
    dm.dummy = 0
    dm.forced = 0 
    # print(f"DataModule")

    
    model_baseline = load_model_student(architecture, baseline=True)
    nets = [get_architecture(architecture), model_baseline, load_model_teacher(architecture)]
            
    if len(nets) == 0:
        raise ValueError("No architecture specified")
    
    elif len(nets) == 1:
        nets = nets[0]
    
    # Si no se especifica versión, seleccionar la nueva para entrenar
    version = len(versions) if args['version'] is None else args['version']

    return args, name, exp_dir, ckpt, version, dm, nets

def get_versions(log_dir, architecture, dataset):
    if isinstance(architecture, list): # Si es una lista, unir los elementos
        architecture = "_".join(architecture)
    experiment_dir = os.path.join(log_dir, f"FastSwimmer_{dataset}")
    if os.path.exists(experiment_dir):
        return os.listdir(experiment_dir)
    else:
        return []
    
def get_experiment(log_dir, architecture, dataset, version=None):
    if isinstance(architecture, list): # Si es una lista, unir los elementos
        architecture = "_".join(architecture)
    experiment_name = f"FastSwimmer_{dataset}"
    experiment_dir = os.path.join(log_dir, experiment_name)
    experiment_version_dir = None
    if version is not None:
        experiment_version_dir = os.path.join(experiment_dir, f"version_{version}", "checkpoints")
        if not os.path.exists(experiment_version_dir): # Verificar si el modelo existe
            raise ValueError(f"Version {version} does not exist")
        else:
            experiment_version_dir = os.path.join(experiment_version_dir, os.listdir(experiment_version_dir)[0])
    return experiment_name, experiment_dir, experiment_version_dir

def get_data_module(dataset, batch_size, data_dir, samples, forced, dummy, device):
    from dataset import MRIDataModule
    dataset_classes = {
        'sagittal': MRIDataModule,
        'axial': MRIDataModule,
        'coronal': MRIDataModule
    }
    try:
        return dataset_classes[dataset](batch_size=batch_size,
                                        data_dir=data_dir,
                                        mode=dataset,
                                        samples=samples,
                                        forced=forced,
                                        dummy=dummy,
                                        device=device)
    except KeyError:
        raise ValueError(f"Invalid dataset: {dataset}")

def get_architecture(architecture):

    
    try:
        return load_model_student(architecture)
    except KeyError:
        raise ValueError(f"Invalid architecture: {architecture}")
    


    

    


