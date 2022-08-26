import os
import torch
import numpy as np
import random
import logging
from tqdm import tqdm

from .video.image_ops import img_from_base64
from .video.video_ops import extract_frames_from_video_binary, extract_frames_from_video_path

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

        self.video_line_list = [i for i in range(self.label_tsv.num_rows())]

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

    def apply_augmentations(self, frames):
        # if failed to decode video, generate fake frames (should be corner case)
        if frames is None:
            frames = np.zeros((self.decoder_num_frames, self.img_res, self.img_res, 3)).astype(np.uint8)
        # (T, C, H, W) -> (T, H, W, C), channel is RGB
        elif 'torch' in str(frames.dtype):
            frames = frames.numpy()
            frames = np.transpose(frames, (0, 2, 3, 1))
        else:
            frames = frames.astype(np.uint8)
            frames = np.transpose(frames, (0, 2, 3, 1))
        num_of_frames, height, width, channels = frames.shape

        frame_list = []
        for i in range(self.decoder_num_frames):
            if num_of_frames == 1:
                # if it is from image-caption dataset, we duplicate the image
                # convert numpy to PIL format, compatible to augmentation operations
                frame_list.append(Image.fromarray(frames[0]))
            else:
                # if it is from video-caption dataset, we add each frame to the list
                # convert numpy to PIL format, compatible to augmentation operations
                frame_list.append(Image.fromarray(frames[i]))

        # adapt from torch_videovision: https://github.com/hassony2/torch_videovision
        # after augmentation, output tensor (C x T x H x W) in the range [0, 1.0]
        crop_frames = self.raw_video_prcoess(frame_list)
        # (C x T x H x W) --> (T x C x H x W)
        crop_frames = crop_frames.permute(1, 0, 2, 3)
        return crop_frames

    def get_video_index(self, idx):
        return self.video_line_list[idx]

    def get_tsv_file(self, tsv_file):
        if tsv_file:
            if self.is_composite:
                return CompositeTSVFile(tsv_file, self.cap_linelist_file, root=self.root)
            tsv_path = find_file_path_in_yaml(tsv_file, self.root)
            return TSVFile(tsv_path)

    def decode_and_get_frames(self, clip_path_name, start=None, end=None):
        # online decode raw video file, and get video frames
        # output tensor (T, C, H, W), channel is RGB, T = self.decoder_num_frames
        if 'TVC' in clip_path_name:
            # default clip_path_name: datasets/TVC/videos/{tv_show}/{tv_show}_clips/{tv_show}_{seasoninfo}/{video_id}.mp4_{start_time}_{end_time}
            # To load video file, we will need to remove start&end info here
            resolved_video_path = '_'.join(clip_path_name.split('_')[0:-2])
        else: # VATEX, MSVD, MSRVTT, Youcook2
            resolved_video_path = clip_path_name
        frames, video_max_pts = extract_frames_from_video_path(resolved_video_path,
                                                self.decoder_target_fps,
                                                self.decoder_num_frames,
                                                self.decoder_multi_thread_decode,
                                                self.decoder_sampling_strategy,
                                                self.decoder_safeguard_duration,
                                                start, end)
        return frames

    def get_visual_data(self, idx, start=None, end=None):
        row = self.get_row_from_tsv(self.visual_tsv, idx)
        # if the input is a video tsv with only video file paths,
        # extract video frames on-the-fly, and return a video-frame tensor
        if row[0] == row[-1]:
            return self.decode_and_get_frames(row[-1], start, end), True
        # if the input is a video tsv with frames pre-extracted,
        # return a video-frame tensor
        elif len(row) >= self.decoder_num_frames + 2:
            return self.get_frames_from_tsv(row[2:]), True
        # if the input is a image tsv, return image numpy array
        else:
            return self.get_image(row[-1]), False

    def __getitem__(self, idx):

        video_idx = self.get_video_index(id)
        # video_key = sekf

        # get image or video frames
        # frames: (T, C, H, W),  is_video: binary tag
        raw_frames, is_video = self.get_visual_data(video_idx, start, end)

        # apply augmentation. frozen-in-time if the input is an image
        # preproc_frames: (T, C, H, W), C = 3, H = W = self.img_res, channel is RGB
        preproc_frames = self.apply_augmentations(raw_frames)

        return preproc_frames

    def __len__(self):
        return self.size
