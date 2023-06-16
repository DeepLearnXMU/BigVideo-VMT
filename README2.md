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

# BigVideo Dataset

Raw videos from [here]() (available soon). 

Dataset are available at [here](https://huggingface.co/datasets/fringek/BigVideo/tree/main). 

```bash
#  structure 
├─ text_data   # original text data and our preprocessed text data
   ├─ test.relate_score  # 1=ambiguous set  0=unambiguous set
   ├─ test.anno.combine  # our annotated ambiguous terms
   ├─ test.id            # refer to corresponding videos
   ......
├─ fairseq_bins # our preprocessed fairseq-bin
├─ video_features # our extracted video features
   ├─ VIT
   ├─ slowfast
```


# Feature Extraction
An example of how to extract VIT features can be seen under fairseq_mmt/scrpits/video_extractor/vit and how to extract frames.tsv can be found in [VideoSwin](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/docs/data_preparation.md#extract-frames).
You can also follow [Hero_extractor](https://github.com/linjieli222/HERO_Video_Feature_Extractor) for more types of video features.

# Train and Test
## Train

To train our model with contrastive learning objective, following arguments are required:
```bash
--arch video_fushion_encoder_revise_one_merge_before_pewln \
--criterion cross_modal_criterion_with_ctr_revise   \
--contrastive-strategy mean+mlp  \
--contrastive-weight ${contrastive_weight}   \
--contrastive-temperature ${contrastive_temperature}  \
--video-feat-path $video_feat_path \
--video-ids-path $video_ids_path \
--video-feat-dim $video_feat_dim \
--video-feat-type $video_feat_type \
--max-vid-len 12  --train-sampling-strategy uniform   \
--video-dropout 0.0  
```
Please check fairseq_mmt/sh/run_video_translation_with_ctr_revise.sh for more details.

## Inference
```bash
fairseq-generate  $test_DATA  \
--path $checkpoint_dir/$checkpoint \
--remove-bpe \
--gen-subset $who \
--beam 4  \
--batch-size  128  \
--lenpen 1.0  \
--video-feat-path $video_feat_path \
--video-ids-path $video_ids_path \
--video-feat-dim $video_feat_dim \
--video-feat-type $video_feat_type \
--max-vid-len $max_vid_len   \
--task raw_video_translation_from_np   | tee $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log

grep ^S $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log | cut -d - -f 2- | sort -n -k 1 | cut -f 2- > $local_output_dir/text.$checkpoint.$length_penalty.$who.src
grep ^H $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log | cut -d - -f 2- | sort -n -k 1 | cut -f 3- > $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo
grep ^T $local_output_dir/text.$checkpoint.$length_penalty.gen-$who.log | cut -d - -f 2- | sort -n -k 1 | cut -f 2- > $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt
```

## Evaluation
### Quality-targeted metrics
To evaluate the generated output, we first need to detokenize the src, hypo, and target
```bash
perl $detokenizer  -l en < $local_output_dir/text.$checkpoint.$length_penalty.$who.src > $local_output_dir/text.$checkpoint.$length_penalty.$who.src.dtk
python3 /root/fairseq_mmt/scripts/detokenize_zh.py --input $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo --output $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo.dtk  # Chinese deokenize
python3 /root/fairseq_mmt/scripts/detokenize_zh.py --input $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt --output $local_output_dir/text.$checkpoint.$length_penalty.$who.tgt.dtk    # Chinese deokenize
```

We evaluate the ouput with [SacreBLEU](https://github.com/mjpost/sacrebleu), [COMET](https://github.com/Unbabel/COMET), and [BLEURT](https://github.com/google-research/bleurt).  
Please check fairseq_mmt/sh/evaluate.py for the whole pipeline.

### Terminology-targetd metrics
We adapt the code from [mahfuzibnalam](https://github.com/mahfuzibnalam/terminology_evaluation) for terminology-targeted evaluation.
You can directly get results like this: 
```bash
bash /PATHTO/terminology_evaluation/test.sh $local_output_dir/text.$checkpoint.$length_penalty.$who.hypo 
```

An example of the whole inference and evaluation pipeline can be found in fairseq_mmt/sh/generate.sh



