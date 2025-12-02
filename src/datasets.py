import torchvision.transforms as transforms
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 64
img_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),  # Scales data into [0,1]
])

class MyDataset(Dataset):
    def __init__(self, root_dir, start_idx, stop_idx):
        self.classes = ["cubes", "spheres"]
        self.root_dir = root_dir
        self.rgb = []
        self.lidar = []
        self.class_idxs = []

        for class_idx, class_name in enumerate(self.classes):
            for idx in range(start_idx, stop_idx):
                file_number = "{:04d}".format(idx)
                rbg_img = Image.open(self.root_dir + class_name + "/rgb/" + file_number + ".png")
                rbg_img = img_transforms(rbg_img).to(device)
                self.rgb.append(rbg_img)
    
                lidar_depth = np.load(self.root_dir + class_name + "/lidar/" + file_number + ".npy")
                lidar_depth = torch.from_numpy(lidar_depth[None, :, :]).to(torch.float32).to(device)
                self.lidar.append(lidar_depth)

                self.class_idxs.append(torch.tensor(class_idx, dtype=torch.float32)[None].to(device))

    def __len__(self):
        return len(self.class_idxs)

    def __getitem__(self, idx):
        rbg_img = self.rgb[idx]
        lidar_depth = self.lidar[idx]
        class_idx = self.class_idxs[idx]
        return rbg_img, lidar_depth, class_idx
    

class ReplicatorDataset(Dataset):
    def __init__(self, root_dir, start_idx, stop_idx):
        self.root_dir = root_dir
        self.rgb_imgs = []
        self.lidar_depths = []
        self.positions = np.genfromtxt(
            root_dir + "positions.csv", delimiter=",", skip_header=1
        )[start_idx:stop_idx]
        

        azimuth = np.load(self.root_dir + "azimuth.npy")
        zenith = np.load(self.root_dir + "zenith.npy")
        self.azimuth = torch.from_numpy(azimuth).to(device)
        self.zenith = torch.from_numpy(zenith).to(device)

        for idx in range(start_idx, stop_idx):
            file_number = "{:04d}".format(idx)
            rbg_img = Image.open(self.root_dir + "rgb/" + file_number + ".png")
            rbg_img = img_transforms(rbg_img).to(device)
            self.rgb_imgs.append(rbg_img)

            lidar_depth = np.load(self.root_dir + "lidar/" + file_number + ".npy")
            lidar_depth = torch.from_numpy(lidar_depth).to(torch.float32).to(device)
            self.lidar_depths.append(lidar_depth)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        rbg_img = self.rgb_imgs[idx]
        lidar_depth = self.lidar_depths[idx]
        lidar_xyza = get_torch_xyza(lidar_depth, self.azimuth, self.zenith)

        position = self.positions[idx]
        position = torch.from_numpy(position).to(torch.float32).to(device)

        return rbg_img, lidar_xyza, position