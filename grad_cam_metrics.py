# Description: 
# - This file contains the implementation of the Knowledge Destillation Model, which is defined
#   using PyTorch Lightning and is utilized for learning on top of the ImageNet dataset.

# 🍦 Vanilla PyTorch
import torch
torch.set_float32_matmul_precision('medium')

from torch import nn
from torch.nn import functional as F

import json

# 📊 TorchMetrics for metrics

# ⚡ PyTorch Lightning
import pytorch_lightning as pl


    
class GradCamLabel(pl.LightningModule):
    def __init__(self, teacher: nn.Module, student: nn.Module, test_dataloader: torch.utils.data.DataLoader, path: str):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.test_dataloader = test_dataloader
        self.path = path

        # Teacher not trainable
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Teacher not in gpu
        self.teacher = self.teacher.to("cpu")

        # Teacher without dropout
        self.teacher.eval()  

        for param in self.student.parameters():
            param.requires_grad = False

        self.student = self.student.to("cpu")
        self.student.eval()
        self.num_classes = self.student.num_classes
    
    def grad_cam_step(self, xs, label):
        heatmaps_teacher, _ = self.teacher.grad_cam(xs, label)
        heatmaps_student, _ = self.student.grad_cam(xs, label)

        return heatmaps_teacher, heatmaps_student
    
    def cosine_similarity(self, heatmaps_teacher, heatmaps_student):
        tensor1 = torch.flatten(heatmaps_teacher)
        tensor2 = torch.flatten(heatmaps_student)

        dot_product = torch.dot(tensor1, tensor2)

        norm1 = torch.norm(tensor1)
        norm2 = torch.norm(tensor2)

        cosine_similarity = dot_product / (norm1 * norm2)

        return cosine_similarity
    
    def loop(self):
        dict_metrics = {}
        length = len(self.test_dataloader().dataset)
        for label in range(self.num_classes):
            coe_total = 0
            for batch in self.test_dataloader():
                xs, _ = batch
                n = xs.size(0)
                heatmaps_teacher, heatmaps_student = self.grad_cam_step(xs, label)
                coe_batch = self.cosine_similarity(heatmaps_teacher, heatmaps_student)
                coe_total += coe_batch * n
            
            coe_total /= length
            dict_metrics[label] = coe_total
            break
        
        self.save(dict_metrics)
    
    def save(self, dict_metrics):
        with open(self.path, 'w') as archivo:
            json.dump(dict_metrics, archivo)


            
if __name__ == "__main__":
    import os
    from utils import get_arguments
    
    # Nombre del experimento
    log_dir = "distiller_logs"
    os.makedirs(log_dir, exist_ok=True)

    args, name, exp_dir, ckpt, version, dm, nets = get_arguments(log_dir, "distiller")
    
    # Cargar el modelo del profesor
    # teacher, student = nets
    student: nn.Module = nets[0]
    baseline: nn.Module = nets[1]
    teacher: nn.Module = nets[2]

    
    # Crear el modelo de destilación
    if ckpt is not None:
        state_dict = torch.load(ckpt, map_location=torch.device('cpu'))["state_dict"]
        keys = state_dict.keys()

        # Eliminar el prefijo "model." de las claves del state_dict
        # new_state_dict = {key.replace("model.", ""): value for key, value in state_dict.items()}
        student.load_state_dict(state_dict)

    else:
        raise ValueError("No checkpoint provided")
    print(ckpt)
    grad_cam = GradCamLabel(teacher, student, dm.test_dataloader, "test.json")
    grad_cam.loop()
    
