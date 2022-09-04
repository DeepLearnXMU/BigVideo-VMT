import os
import torch
import numpy as np
import random
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VideoDatasetFromNp(torch.utils.data.Dataset):
    """
    For loading image datasets
    """

    def __init__(self, args,split):
        self.split=split
        self.video_feat_type=args.video_feat_type
        self.video_feat_path = args.video_feat_path
        self.video_ids_path = args.video_ids_path
        self.max_vid_len = args.max_vid_len
        self.video_path = os.path.join(self.video_feat_path,self.split)

        self.video_id_list = []
        with open(self.video_ids_path + "/" + split + ".id", encoding='utf-8') as file:
            self.video_id_list = [x.rstrip() for x in file.readlines()]

        self.video_feat_type = args.video_feat_type
        self.video_feat_dim = args.video_feat_dim

        self.size = len(self.video_id_list)

    def __getitem__(self, idx):

        video_name = self.video_id_list[idx]
        fpath = os.path.join(self.video_path ,video_name+".npz")
        features = np.load(fpath, encoding='latin1')["features"]
        return features

    def __len__(self):
        return self.size
