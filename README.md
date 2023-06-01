This repository is for the accepted ACL2023 Findings paper 
"[BIGVIDEO: A Large-scale Video Subtitle Translation Dataset for Multimodal Machine Translation](https://arxiv.org/abs/2305.18326)". 

# Our dependency

* PyTorch version == 1.10.0
* timm version == 0.4.12
* vizseq version == 0.1.15
* nltk verison == 3.6.4
* sacrebleu version == 1.5.1
* Please check fairseq_mmt/sh/requirements.txt for more details

# Install fairseq

```bash
cd fairseq_mmt
pip install --editable ./
```

# BigVideo-data
## Video
Raw videos from [here]() (available soon). 

Our extracted 2d features from [here]() (available soon) and 3d features
from [here]()  (available soon).


An example of how to extract VIT features can be seen under fairseq_mmt/scrpits/video_extractor/vit/

For 3D features, please follow [hero](https://github.com/linjieli222/HERO_Video_Feature_Extractor).

```bash
# Final data structure like this
data
├─ train
├─ valid
├─ test
├─ fairseq_bin
├─ video_features
   ├─ VIT
   ├─ slowfast
```



# Train and Test
Todo

