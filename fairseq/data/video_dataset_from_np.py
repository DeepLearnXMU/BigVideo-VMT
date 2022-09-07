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

        # for no video , pad one
        self.video_pad = np.random.RandomState(0).normal(loc=0.0, scale=1, size=(video_feat_dim), )


    def __getitem__(self, idx):

        video_name = self.video_id_list[idx]
        fpath = os.path.join(self.video_path ,video_name+".npz")
        empty_flag = False
        if not os.path.exists(fpath):
            features = np.zeros((1, self.video_feat_dim))
            empty_flag = True
        else:
            features = np.load(fpath, encoding='latin1')["features"]

        padding = np.zeros(self.max_vid_len)
        if empty_flag:
            features[0] = self.video_pad
            padding[0] = 0
            padding[1:] = 1
        else:
            if features.shape[0] < self.max_vid_len:
                dis = self.max_vid_len - features.shape[0]
                padding[features.shape[0]:] = 1
                features = np.lib.pad(features, ((0, dis), (0, 0)), 'constant', constant_values=0)
            elif features.shape[0] > max_length:
                inds = sorted(random.sample(range(features.shape[0]), max_length))
                features = features[inds]
        return features,padding

    def __len__(self):
        return self.size
