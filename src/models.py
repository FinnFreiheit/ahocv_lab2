import torch
import torch.nn as nn
import torch.nn.functional as F

# Please do no alter this file. That will make it harder to pass the assessment!
class Classifier(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        kernel_size = 3
        n_classes = 1
        self.embedder = nn.Sequential(
            nn.Conv2d(in_ch, 50, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(50, 100, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(100, 200, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(200, 200, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(200 * 4 * 4, 100),
            nn.ReLU(),
            nn.Linear(100, n_classes)
        )

    def get_embs(self, imgs):
        return self.embedder(imgs)
    
    def forward(self, raw_data=None, data_embs=None):
        assert (raw_data is not None or data_embs is not None), "No images or embeddings given."
        if raw_data is not None:
            data_embs = self.get_embs(raw_data)
        return self.classifier(data_embs)
    

class Embedder(nn.Module):
    def __init__(self, in_ch, emb_size=200):
        super().__init__()
        kernel_size = 3
        stride = 1
        padding = 1

        # Convolution
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 50, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(50, 100, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(100, 200, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(200, 200, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        # Embeddings
        self.dense_emb = nn.Sequential(
            nn.Linear(200 * 4 * 4, 100),
            nn.ReLU(),
            nn.Linear(100, emb_size)
        )

    def forward(self, x):
        conv = self.conv(x)
        emb = self.dense_emb(conv)
        return F.normalize(emb)
    

class ContrastivePretraining(nn.Module):
    def __init__(self, img_embedder, lidar_embedder):
        super().__init__()
        self.img_embedder = img_embedder
        self.lidar_embedder = lidar_embedder
        self.cos = nn.CosineSimilarity()

    def forward(self, rgb_imgs, lidar_depths):
        img_emb = self.img_embedder(rgb_imgs)
        lidar_emb = self.lidar_embedder(lidar_depths)

        repeated_img_emb = img_emb.repeat_interleave(len(img_emb), dim=0)
        repeated_lidar_emb = lidar_emb.repeat(len(lidar_emb), 1)

        similarity = self.cos(repeated_img_emb, repeated_lidar_emb)
        similarity = torch.unflatten(similarity, 0, (32, 32))
        similarity = (similarity + 1) / 2

        logits_per_img = similarity
        logits_per_lidar = similarity.T
        return logits_per_img, logits_per_lidar
    
class RGB2LiDARClassifier(nn.Module):
    def __init__(self, projector, CILP_model, lidar_cnn):
        super().__init__()
        self.projector = projector
        self.img_embedder = CILP_model.img_embedder
        self.shape_classifier = lidar_cnn
    
    def forward(self, imgs):
        img_encodings = self.img_embedder(imgs)
        proj_lidar_embs = self.projector(img_encodings)
        return self.shape_classifier(data_embs=proj_lidar_embs)