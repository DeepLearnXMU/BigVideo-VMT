import os
import torch
import numpy as np

def load_video_features(fpath, max_length):
    feats = np.load(fpath, encoding='latin1')[0]  # encoding='latin1' to handle the inconsistency between python 2 and 3
    padding = np.ones(max_length)
    if feats.shape[0] < max_length:
        dis = max_length - feats.shape[0]
        padding[feats.shape[0]:] = 0
        feats = np.lib.pad(feats, ((0, dis), (0, 0)), 'constant', constant_values=0)
    elif feats.shape[0] > max_length:
        inds = sorted(random.sample(range(feats.shape[0]), max_length))
        feats = feats[inds]
    assert feats.shape[0] == max_length
    # mask = np.array(padding, dtype=bool)
    return np.float32(feats), padding


class VideoDataset(torch.utils.data.Dataset):
    """
    For loading image datasets
    """
    def __init__(self, video_feat_path: str, video_ids_path: str, max_vid_len:int,split:str):
        self.video_feat_path = video_feat_path
        self.video_ids_path = video_ids_path
        self.max_vid_len = max_vid_len

        self.sent_id_list=[]
        with open(video_ids_path+"/"+split+".ids", encoding='utf-8') as file:
            self.sent_id_list = [x.rstrip() for x in file.readlines()]


    def __getitem__(self, idx):
        sent_id = self.sent_id_list[idx]
        vid = sent_id[:-2]
        img,padding = load_video_features(os.path.join(self.data_dir, 'vatex_features/', self.img_dir, vid + '.npy'),
                                  self.max_vid_len)
        return img

    def __len__(self):
        return self.size
