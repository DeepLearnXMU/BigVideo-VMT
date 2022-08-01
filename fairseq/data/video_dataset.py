import os
import torch
import numpy as np
import random


class VideoDataset(torch.utils.data.Dataset):
    """
    For loading image datasets
    """

    def __init__(self, video_feat_path: str, video_ids_path: str, max_vid_len: int, split: str, video_feat_type: str, ):
        self.video_feat_path = video_feat_path
        self.video_ids_path = video_ids_path
        self.max_vid_len = max_vid_len

        self.sent_id_list = []
        with open(video_ids_path + "/" + split + ".ids", encoding='utf-8') as file:
            self.sent_id_list = [x.rstrip() for x in file.readlines()]

        self.video_feat_type = video_feat_type
        if video_feat_type == "I3D":
            if split in ["train", "valid"]:
                self.video_dir = "trainval/"
            else:
                self.video_dir = "public_test/"
        elif video_feat_type == "VIT_cls":
            self.video_dir = "cls/" + split

        elif video_feat_type == "VIT_patch":
            self.video_dir = "patch/" + split

        self.video_list = []
        self.padding_list = []

        for sent_id in self.sent_id_list:
            vid = sent_id[:-2]
            video, padding = self.load_video_features(os.path.join(self.video_feat_path, self.video_dir, vid + '.npy'),
                                                      self.max_vid_len)
            self.video_list.append(video)
            self.padding_list.append(padding)

        assert (len(self.video_list) == len(self.sent_id_list))
        self.size = len(self.sent_id_list)

    def load_video_features(self, fpath, max_length):
        feats = np.array([])
        empty_flag = False
        if not os.path.exists(fpath):
            feats = np.zeros((1, 768))
            empty_flag = True
        elif self.video_feat_type == "I3D":
            feats = np.load(fpath, encoding='latin1')[0]  # encoding='latin1' to handle the inconsistency between python 2 and 3
        elif self.video_feat_type in ["VIT_cls", "VIT_cls"]:
            feats = np.load(fpath, encoding='latin1')
        padding = np.ones(max_length)
        if feats.shape[0] < max_length:
            dis = max_length - feats.shape[0]
            padding[feats.shape[0]:] = 0
            feats = np.lib.pad(feats, ((0, dis), (0, 0)), 'constant', constant_values=0)
        elif feats.shape[0] > max_length:
            inds = sorted(random.sample(range(feats.shape[0]), max_length))
            feats = feats[inds]
        if empty_flag:
            padding[:] = 0
        assert feats.shape[0] == max_length
        # mask = np.array(padding, dtype=bool)
        return np.float32(feats), padding

    def __getitem__(self, idx):

        return self.video_list[idx], self.padding_list[idx]

    def __len__(self):
        return self.size
