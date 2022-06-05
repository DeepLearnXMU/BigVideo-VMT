import os
import torch

class ImageDataset(torch.utils.data.Dataset):
    """
    For loading image datasets
    """
    def __init__(self, feat_path: str, mask_path: str):
        self.img_feat = torch.load(feat_path)
        self.img_feat_mask = None
        if os.path.exists(mask_path):
            self.img_feat_mask = torch.load(mask_path)

        self.size = self.img_feat.shape[0]

    def __getitem__(self, idx):
        if self.img_feat_mask is None:
            return self.img_feat[idx], None
        else:
            return self.img_feat[idx], self.img_feat_mask[idx]

    def __len__(self):
        return self.size
