import os
import torch
import numpy as np
import random
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (list(av.video.frame.VideoFrame)): a list of decoded video frames
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, len(frames) - 1).long().tolist()
    return frames[index]


def rand_sampling(frames, start_idx, end_idx, num_samples):
    if end_idx - start_idx + 1 < num_samples:
        return frames[start_idx:end_idx + 1]
    else:
        inds = sorted(random.sample(range(start_idx, end_idx + 1), num_samples))
        return frames[inds]


def continuous_sampling(frames, start_idx, end_idx, num_samples):
    if end_idx - start_idx + 1 < num_samples:
        return frames[start_idx:end_idx + 1]
    else:
        choose_index = random.randint(start_idx, end_idx - num_samples + 1)
        index = range(choose_index, choose_index + num_samples)
        return frames[index]


class VideoDatasetFromNp(torch.utils.data.Dataset):
    """
    For loading image datasets
    """

    def __init__(self, args, split):
        self.split = split
        self.video_feat_type = args.video_feat_type
        self.video_feat_path = args.video_feat_path
        self.video_ids_path = args.video_ids_path
        self.max_vid_len = args.max_vid_len
        self.video_path = os.path.join(self.video_feat_path, self.split)
        self.id_type = args.id_type

        self.video_id_list = []

        if self.id_type == "original":
            id_file=f"{self.video_ids_path}/{split}.id"
        else:
            id_file = f"{self.video_ids_path}/{split}.{self.id_type}.id"
            assert os.path.exists(id_file)

        with open(id_file, encoding='utf-8') as file:
            self.video_id_list = [x.rstrip() for x in file.readlines()]

        self.video_feat_type = args.video_feat_type
        self.video_feat_dim = args.video_feat_dim

        self.size = len(self.video_id_list)

        self.sampling_strategy = args.sampling_strategy
        self.sampling_frames = args.sampling_frames
        if self.sampling_strategy == ("rand" or "uniform"):
            assert self.sampling_frames > 0

        # for no video , pad one
        self.video_pad = np.random.RandomState(0).normal(loc=0.0, scale=1, size=(args.video_feat_dim), )

    def __getitem__(self, idx):

        video_name = self.video_id_list[idx]
        fpath = os.path.join(self.video_path, video_name + ".npz")
        empty_flag = False
        if not os.path.exists(fpath):
            features = np.zeros((1, self.video_feat_dim))
            empty_flag = True
        else:
            features = np.load(fpath, encoding='latin1')["features"]
            if len(features.shape) == 3:
                features = features[0]

        start_idx, end_idx = 0, features.shape[0] - 1

        if self.sampling_strategy == "None" or self.split != "train":
            padding = np.zeros(self.max_vid_len)
            if features.shape[0] < self.max_vid_len:
                dis = self.max_vid_len - features.shape[0]
                padding[features.shape[0]:] = 1
                features = np.lib.pad(features, ((0, dis), (0, 0)), 'constant', constant_values=0)
            elif features.shape[0] > self.max_vid_len:
                inds = sorted(random.sample(range(features.shape[0]), self.max_vid_len))
                features = features[inds]
            if empty_flag:
                features[0] = self.video_pad
                padding[0] = 0
                padding[1:] = 1
            assert features.shape[0] == self.max_vid_len

            return features, padding
        else:
            if empty_flag:
                dis = self.sampling_frames - features.shape[0]
                features = np.lib.pad(features, ((0, dis), (0, 0)), 'constant', constant_values=0)
                padding = np.zeros(self.sampling_frames)
                features[0] = self.video_pad
                padding[0] = 0
                padding[1:] = 1
            elif features.shape[0] < self.sampling_frames:
                dis = self.sampling_frames - features.shape[0]
                padding = np.zeros(self.sampling_frames)
                features = np.lib.pad(features, ((0, dis), (0, 0)), 'constant', constant_values=0)
                padding[features.shape[0]:] = 1
            else:
                padding = np.zeros(self.sampling_frames)
                if self.sampling_strategy == "rand":
                    features = rand_sampling(features, start_idx, end_idx, self.sampling_frames)
                elif self.sampling_strategy == "uniform":
                    features = temporal_sampling(features, start_idx, end_idx, self.sampling_frames)
                elif self.sampling_strategy == "continuous":
                    features = continuous_sampling(features, start_idx, end_idx, self.sampling_frames)

            assert features.shape[0] == self.sampling_frames
            return features, padding

    def __len__(self):
        return self.size
