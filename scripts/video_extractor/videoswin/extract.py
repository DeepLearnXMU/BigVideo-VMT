from typing_extensions import dataclass_transform
import torch as th
import math
import numpy as np
# from video_loader import VideoLoader
from torch.utils.data import DataLoader
import argparse
# from model import get_model
# from preprocessing import Preprocessing
import torch.nn.functional as F
from tqdm import tqdm
import os
from raw_video_dataset import RawVideoDataset
import timm
import time
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Sampler
from fairseq.models.video.load_swin import get_swin_model, reload_pretrained_swin

parser = argparse.ArgumentParser(description='Easy video feature extractor')

parser.add_argument(
    '--csv',
    type=str,
    help='input csv with video input path')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--type', type=str, default='2d',
                    choices=["2d"],
                    # 3d is also supported in original repo,
                    # we disable here as we are using slowfast
                    help='CNN type')
parser.add_argument(
    '--clip_len', type=float, default=3 / 2,
    help='decoding length of clip (in seconds)')
parser.add_argument(
    '--overwrite', action='store_true',
    help='allow overwrite output files')
parser.add_argument('--half_precision', type=int, default=1,
                    help='output half precision float')
parser.add_argument('--num_decoding_thread', type=int, default=8,
                    help='Num parallel thread for video decoding')
parser.add_argument('--l2_normalize', type=int, default=1,
                    help='l2 normalize feature')
parser.add_argument('--resnext101_model_path', type=str,
                    default='./3d_resnets/model/resnext101.pth',
                    help='Resnext model path')

parser.add_argument('--split', type=str)
parser.add_argument('--visual-dir', type=str)
parser.add_argument('--video-feat-type', type=str)
parser.add_argument('--video-feat-dim', type=int)
parser.add_argument('--max-num-frames', type=int, default=32)
parser.add_argument('--img-res', type=int, default=224)
parser.add_argument('--patch-size', type=int, default=32)
parser.add_argument('--output-dir', type=str)
parser.add_argument('--choice', type=str)
parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
parser.add_argument('--shard_num', type=int)
parser.add_argument('--shard_id', type=int)
parser.add_argument('--videoswin-path', type=str)
parser.add_argument("--kinetics", type=str, default='600', help="400 or 600")
parser.add_argument("--pretrained-2d", type=bool, nargs='?', const=True, default=False)
parser.add_argument("--vidswin-size", type=str, default='base')

args = parser.parse_args()

# if args.local_rank != -1:
#     torch.cuda.set_device(args.local_rank)
#     device=torch.device("cuda", args.local_rank)
#     torch.distributed.init_process_group(backend="nccl", init_method='env://')

args.video_feat_type = "2d"
assert args.type == '2d', "Only 2d feature extraction is tested under this release"

split = args.split
dataset = RawVideoDataset(
    args, split
)

n_dataset = len(dataset)

start_index = (n_dataset // args.shard_num) * args.shard_id
if args.shard_id == args.shard_num - 1:
    end_index = n_dataset
else:
    end_index = (n_dataset // args.shard_num) * (args.shard_id + 1)

print(dataset[start_index]["name"], dataset[start_index - 1]["name"], dataset[start_index + 1]["name"])
print(f"from {start_index} to {end_index}")
index_list = list(range(start_index, end_index))


# sampler = DistributedSampler(dataset)
# sampler = SequentialSampler(index_list)

class ShardSampler(Sampler):
    def __init__(self, index_list):
        self.index_list = index_list

    def __iter__(self):
        return iter(self.index_list)

    def __len__(self):
        return len(self.index_list)


sampler = ShardSampler(index_list)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args.num_decoding_thread,
    sampler=sampler
)

model = get_swin_model(args)


# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     model = torch.nn.DataParallel(model,device_ids=range(torch.cuda.device_count()))


# model =  clip.load("ViT-B/16", device="cuda")

output_dir = os.path.join(args.output_dir, split)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
print("saving to", output_dir)
totatl_num_frames = 0

with th.no_grad():
    shard_names = []
    shard_features = []

    for k, data in enumerate(tqdm(loader)):

        if k == 0:
            print(data["name"][0], dataset[start_index]["name"])
            assert data["name"][0] == dataset[start_index]["name"]

        video, name = data["video"][0], data["name"][0]
        video = video.cuda()
        # print(video.shape)

        # if len(video.shape) > 4:
        #     video = video.view(-1,3,224,224)

        batch_features = model.forward_features(video, choice=args.choice)
        # print(batch_features.shape)

        if args.choice == "patch":
            out = th.mean(batch_features, dim=0)

        output_file = os.path.join(output_dir, name + ".npz")
        now_feature = batch_features.cpu().numpy()
        if args.half_precision:
            now_feature = now_feature.astype('float16')
        np.savez(output_file, features=now_feature)



print(f"Total number of frames: {totatl_num_frames}")
