import os
import torch
import numpy as np
import random
import logging
from tqdm import tqdm

# video_transforms & volume_transforms from https://github.com/hassony2/torch_videovision
from .video.video_transforms import Compose, Resize, RandomCrop, ColorJitter, Normalize, CenterCrop, RandomHorizontalFlip, RandomResizedCrop
from .video.volume_transforms import ClipToTensor

logger = logging.getLogger(__name__)


class RawVideoDataset(torch.utils.data.Dataset):
    """
    For loading image datasets
    """

    def __init__(self, args):

        self.visual_file = getattr(args, 'visual_file', None)
        assert  self.visual_file is not None
        self.visual_tsv = self.get_tsv_file(self.visual_file)

        self.is_train = (split=="Train")
        self.patch_size = getattr(args, 'patch_size', 16)

        self.img_feature_dim = args.img_feature_dim
        self.decoder_target_fps = 3
        self.decoder_num_frames = getattr(args, 'max_num_frames', 2)
        self.decoder_multi_thread_decode = False

        self.decoder_safeguard_duration = False
        self.add_od_labels = getattr(args, 'add_od_labels', False)
        self.use_asr = getattr(args, 'use_asr', False)

        self.decoder_sampling_strategy = getattr(args, 'decoder_sampling_strategy', 'uniform')
        logger.info(f'isTrainData: {self.is_train}\n[video parameters] '
                    f'Num of Frame: {self.decoder_num_frames}, '
                    f'FPS: {self.decoder_target_fps}, '
                    f'Sampling: {self.decoder_sampling_strategy}')

        if self.is_train:
            self.raw_video_crop_list = [
                Resize(self.img_res),
                RandomCrop((self.img_res, self.img_res)),
                ClipToTensor(channel_nb=3),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        else:
            self.raw_video_crop_list = [
                Resize(self.img_res),
                CenterCrop((self.img_res, self.img_res)),
                ClipToTensor(channel_nb=3),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        self.raw_video_process = Compose(self.raw_video_crop_list)

    def get_tsv_file(self, tsv_file):
        if tsv_file:
            if self.is_composite:
                return CompositeTSVFile(tsv_file, self.cap_linelist_file, root=self.root)
            tsv_path = find_file_path_in_yaml(tsv_file, self.root)
            return TSVFile(tsv_path)

    def __getitem__(self, idx):

        return self.video_list[idx], self.padding_list[idx]

    def __len__(self):
        return self.size
