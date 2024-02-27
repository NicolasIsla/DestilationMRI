import os
import gdown
import zipfile
import gzip
import nibabel as nib
import shutil
import numpy as np

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

import utils

class Preprocess:
    def __init__(self, data_dir, mode, samples=10, forced=False, dummy=False):
        self.data_dir = data_dir
        self.samples = samples
        self.mode = mode
        self.forced = forced
        self.dummy = dummy
        if self.forced:
            # remove all folder in the data_dir
            for folder in os.listdir(self.data_dir):
                if os.path.isdir(f"{self.data_dir}{folder}"):
                    shutil.rmtree(f"{self.data_dir}{folder}")

            for file in os.listdir(self.data_dir):
                os.remove(f"{self.data_dir}{file}")
            

    def dowload_model(self):
        if not os.path.exists(f"./checkpoints") or self.forced:
            
            id = "1joAF85z_K2tg7ZRUVK7rWRWrR6aFkZ9q"
            gdown.download(f"https://drive.google.com/uc?id={id}", f"./model.zip", quiet=False)  
            # unzip the model
            with zipfile.ZipFile(f"./model.zip", 'r') as zip_ref:
                zip_ref.extractall(f"./")
            # remove the zip file
            os.remove(f"./model.zip")
        else:
            pass
    def download_data(self):
        if not os.path.exists(f"{self.data_dir}train") or self.forced:
            # train
            id = "1vOklcxOINBYSYU-muWoP0FTE0jr9xjpx"
            gdown.download(f"https://drive.google.com/uc?id={id}", f"{self.data_dir}train.zip", quiet=False)
            # val
            id ="1ena233BRJKvIlYBSe-CciSYNjiC3I5A9"
            gdown.download(f"https://drive.google.com/uc?id={id}", f"{self.data_dir}val.zip", quiet=False)
            # test
            id = "1gU6S3GQ56_tUxzkn6_9V7fdD7cfaQf0r"
            gdown.download(f"https://drive.google.com/uc?id={id}", f"{self.data_dir}test.zip", quiet=False)
        else:
            pass

    def download_data_dummy(self):
        if not os.path.exists(f"{self.data_dir}train") or self.forced:
            id = "1kSQrRDZ_9aobn_mO9_yD1CRRr800jjLV"
            gdown.download(f"https://drive.google.com/uc?id={id}", f"{self.data_dir}train.zip", quiet=False)
            id = "1w8VIgFlSICZ-NKHxTNMZ-Qfu353GRgGN"
            gdown.download(f"https://drive.google.com/uc?id={id}", f"{self.data_dir}val.zip", quiet=False)   
            id = "1H6DQB7BCkrG77IS_QMTypeLCWb4LyKLA"
            gdown.download(f"https://drive.google.com/uc?id={id}", f"{self.data_dir}test.zip", quiet=False)    


    def unzip_data(self, folder_path):
        with zipfile.ZipFile(folder_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)


    def ungzip_data(self, file_path):
        with gzip.open(file_path, 'rb') as f, open(file_path.split(".gz")[0], 'wb') as f_out:
            shutil.copyfileobj(f, f_out)

            
    def read_nii(self, file_path):
        data = nib.load(file_path)
        return data.get_fdata()
    
    def padding(self, data, size=256):
        """
        Add padding to the data to make it 256x256
        """
        padding_needed = size - data.shape[1]
        data = np.pad(data, ((0,0), (0, padding_needed), (0,0)), 'constant', constant_values=0)
        padding_needed = size - data.shape[2]
        data = np.pad(data, ((0,0), (0,0), (0, padding_needed)), 'constant', constant_values=0)
        return data

    def create_packages(self, data):
        center = len(data) // 2
        examples = np.clip(
                    np.random.normal(loc=center, scale=center//6, size=self.samples).astype(int),
                    3, len(data)-4)
        out = np.zeros((self.samples, 7, 256, 256))
        
        for i, example in enumerate(examples):
            slice = data[example-3:example+4]
            # in case of error in the dimensions
            if slice.shape[1] != 256 or slice.shape[2] != 256:
                slice = self.padding(slice)
            out[i] = slice
        return out
    
    def mode_packages(self, data):  
        if self.mode == "sagittal":
            # 90° rotation
            return np.rot90(self.create_packages(data), axes=(2, 3))
        elif self.mode == "axial":
            data = np.swapaxes(data, 0, 2)
            return self.create_packages(data)
        elif self.mode == "coronal": 
            data = np.swapaxes(data, 0, 1)
            # 90° rotation
            return np.rot90(self.create_packages(data), axes=(2,3))
        else:
            raise ValueError("Mode not found")


    def preprocess(self):
        self.dowload_model()

        if self.dummy:
            self.download_data_dummy()

        else:
            self.download_data()
        if not os.path.exists(f"{self.data_dir}train.npy") or self.forced:
            for folder in os.listdir(self.data_dir):
                if folder.endswith(".zip"):
                    self.unzip_data(f"{self.data_dir}{folder}")
                    folder = folder.split(".")[0]
                    # number of samples in the folder it
                    n = len(os.listdir(f"{self.data_dir}{folder}"))*self.samples
                    data = np.zeros((n, 7, 256, 256))
                    for i, file in enumerate(os.listdir(f"{self.data_dir}{folder}")):
                        if file.endswith(".gz"):
                            self.ungzip_data(f"{self.data_dir}{folder}/{file}")
                            # read the nii file
                            file_nii = file.split(".gz")[0]
                            data_mri = self.read_nii(f"{self.data_dir}{folder}/{file_nii}")
                            os.remove(f"{self.data_dir}{folder}/{file_nii}")

                            # create the packages
                            packages = self.mode_packages(data_mri)
                            
                            data[i*self.samples:(i+1)*self.samples] = packages

                    
            
                    # save the data
                    np.save(f"{self.data_dir}{folder}.npy", data)
                    # print(data.shape)

    def create_labels(self, device):
        self.device = torch.device(device)
        model = utils.load_model_teacher(self.mode).eval().to(self.device)
        for file in os.listdir(self.data_dir):
            if file.endswith(".npy"):
                data = np.load(f"{self.data_dir}{file}")
                data_loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=2, persistent_workers=True, pin_memory=True)
                
                labels = []
                for batch in data_loader:
                    with torch.no_grad():
                        output = model.forward_label(batch.float().to(self.device))
                        labels.append(output.cpu().numpy())  
                labels = np.concatenate(labels, axis=0)  
                # print(labels.shape)
                np.save(f"{self.data_dir}{file.split('.')[0]}_labels.npy", labels)
        

class MRIDataset(Dataset):
    def __init__(self, data_dir, split):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.data = torch.from_numpy(np.load(f"{self.data_dir}{self.split}.npy")) 
        self.labels = torch.from_numpy(np.load(f"{self.data_dir}{self.split}_labels.npy"))

        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def __repr__(self):
        return f"ImageNet Dataset: {self.split} split"
    
    def __str__(self):
        return f"ImageNet Dataset: {self.split} split"
    

class MRIDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, mode, batch_size=32,device="cpu",  samples=10, forced=False, dummy=False):
        super().__init__()
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.mode = mode
        self.batch_size = batch_size
        self.samples = samples
        self.forced = forced
        self.dummy = dummy
        self.device = device
    
        
    
    def setup(self):
        preprocess = Preprocess(self.data_dir, self.mode, self.samples, self.forced, self.dummy)
        preprocess.preprocess()
        preprocess.create_labels(self.device)

        self.train = MRIDataset(self.data_dir, "train")
        self.val = MRIDataset(self.data_dir, "val")
        self.test = MRIDataset(self.data_dir, "test")
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=2, persistent_workers=True, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=2, persistent_workers=True, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=2, persistent_workers=True, pin_memory=True)
        


    



        

if __name__ == "__main__":
    data_dir = "D:/data3/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


    modes =["sagittal", "axial", "coronal"]
    for i in modes:
        mriDataModule = MRIDataModule(data_dir, i, batch_size=32, device="cpu", samples=1, forced=True, dummy=True)

    

