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
from tqdm import tqdm


    
class GradCamLabel(pl.LightningModule):
    def __init__(self, teacher: nn.Module, student: nn.Module, test_dataloader: torch.utils.data.DataLoader , path: str, device):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.test_dataloader = test_dataloader
        self.device_aux = torch.device(device)
        
        self.path = os.path.join(path.split["/"][:-2], "metrics")
        if not os.path.exists(self.path):
            os.makedirs(self.path)



        # Teacher not in gpu
        self.teacher = self.teacher.to(self.device_aux)

        # Teacher without dropout
        self.teacher.eval()  


        self.student = self.student.to(self.device_aux)
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

        value = dot_product / (norm1 * norm2)

        return value.item()
    
    def loop(self):
        dict_metrics = {}
        length = len(self.test_dataloader().dataset)
        progress_bar = tqdm(total=self.num_classes, desc="GradCam per class")

        for label in range(self.num_classes):
            coe_total = 0
            for batch in self.test_dataloader():
                xs, _ = batch
                xs = xs.to(self.device_aux)
                heatmaps_teacher, heatmaps_student = self.grad_cam_step(xs, label)
                coe_batch = self.cosine_similarity(heatmaps_teacher, heatmaps_student)
                coe_total += coe_batch 
            
            coe_total /= length
            dict_metrics[label] = coe_total
            progress_bar.update(1)
        
        self.save(dict_metrics)
    
    def save(self, dict_metrics):
        with open(os.path.join(self.path,"metrics.txt"), 'w') as archivo:
            json.dump(dict_metrics, archivo)


            
if __name__ == "__main__":
    import os
    from utils import get_arguments_metrics
    
    # Nombre del experimento
    log_dir = "distiller_logs"
    os.makedirs(log_dir, exist_ok=True)

    args, name, exp_dir, ckpt, version, dm, nets = get_arguments_metrics(log_dir, "distiller")
    
    # Cargar el modelo del profesor
    # teacher, student = nets
    student: nn.Module = nets[0]
    baseline: nn.Module = nets[1]
    teacher: nn.Module = nets[2]
    print(args.device)

    
    # Crear el modelo de destilación
    if ckpt is not None:
        state_dict = torch.load(ckpt, map_location=torch.device('cpu'))["state_dict"]
        keys = state_dict.keys()
        # drop teacher weights
        keys = [key for key in keys if "teacher" not in key] 
        
        # Eliminar el prefijo "model." de las claves del state_dict
        new_state_dict = {key.replace("student.", ""): state_dict[key]  for key in keys}
        
        student.load_state_dict(new_state_dict)

    else:
        raise ValueError("No checkpoint provided")
    
    grad_cam = GradCamLabel(teacher, student, dm.test_dataloader, ckpt, device=args.device)
    grad_cam.loop()
    
