from utils import *
import numpy as np


def cosine_similarity(data_1, data_2):

    vector1 = data_1.flatten()
    vector2 = data_2.flatten()

    dot_product = torch.dot(vector1, vector2)

    norm1 = torch.norm(vector1)
    norm2 = torch.norm(vector2)

    cosine_similarity = dot_product / (norm1 * norm2)

    return cosine_similarity

if __name__ == "__main__":
    import os
    from fastsurfer.networks import *
    import argparse

    dir = "./data/"
    log_dir = "trainer_logs"

    parser = argparse.ArgumentParser(description='Load a model')
    parser.add_argument('--mode', type=str, choices=['sagittal', 'axial','coronal'], help='Mode to be loaded')
    parser.add_argument('--version', type=int, default=None, help='Select a version to load from')
    parser.add_argument('--show_versions', type=int, default=1, help='Show available versions to load from')
    parser.add_argument('--device', type=int, default=0, help='Device to use for training')
    
    args = parser.parse_args()

    architecture = args["mode"]
    
    versions = get_versions(log_dir, architecture, args['model'])

    if args["show_versions"]:
        print(f"Versions: {versions}")
        exit(0)

    model_student = load_model_student(architecture)
    model_teacher = load_model_teacher(architecture)

    # Definir el path del archivo .ckpt en train_logs
    path = os.path.join(log_dir, f"{architecture}_version={args['version']}.ckpt")

    state_dict = torch.load(path, map_location=torch.device('cpu'))["state_dict"]
    keys = state_dict.keys()

    # Eliminar el prefijo "model." de las claves del state_dict
    new_state_dict = {key.replace("model.", ""): value for key, value in state_dict.items()}
    model_student.load_state_dict(new_state_dict)

    # cargar data test
    path = os.path.join(dir, args["mode"], "test.npy")
    input = np.load(path)
    i = 70
    example = input[i:i+1]
    example = torch.from_numpy(example)
    example = example.float()

    example = example.to("cpu")