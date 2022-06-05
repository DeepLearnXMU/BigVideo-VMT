#! /usr/bin/bash
set -e

device=0,1
task=multi30k-en2de
mask_data=mask0

if [ ! -d $save_dir ]; then
        mkdir -p $save_dir
fi

if [ $task == 'multi30k-en2de' ]; then
	src_lang=en
	tgt_lang=de
	if [ $mask_data == "mask0" ]; then
        	data_dir=multi30k.en-de
	elif [ $mask_data == "mask1" ]; then
	        data_dir=multi30k.en-de.mask1
	elif [ $mask_data == "mask2" ]; then
      		data_dir=multi30k.en-de.mask2
	elif [ $mask_data == "mask3" ]; then
	        data_dir=multi30k.en-de.mask3
	elif [ $mask_data == "mask4" ]; then
	        data_dir=multi30k.en-de.mask4
        elif [ $mask_data == "maskc" ]; then
	        data_dir=multi30k.en-de.maskc
        elif [ $mask_data == "maskp" ]; then
	        data_dir=multi30k.en-de.maskp
	fi
elif [ $task == 'multi30k-en2fr' ]; then
	src_lang=en
	tgt_lang=fr
	if [ $mask_data == "mask0" ]; then
        	data_dir=multi30k.en-fr
	elif [ $mask_data == "mask1" ]; then
	        data_dir=multi30k.en-fr.mask1
	elif [ $mask_data == "mask2" ]; then
      		data_dir=multi30k.en-fr.mask2
	elif [ $mask_data == "mask3" ]; then
	        data_dir=multi30k.en-fr.mask3
	elif [ $mask_data == "mask4" ]; then
	        data_dir=multi30k.en-fr.mask4
        elif [ $mask_data == "maskc" ]; then
	        data_dir=multi30k.en-fr.maskc
        elif [ $mask_data == "maskp" ]; then
	        data_dir=multi30k.en-fr.maskp
	fi
fi

criterion=label_smoothed_cross_entropy
fp16=1 #0
lr=0.005
warmup=2000
max_tokens=4096
update_freq=1
keep_last_epochs=10
patience=10
max_update=8000
dropout=0.3
seed=1
arch=transformer_tiny

name=baseline_arch${arch}_tgt${tgt_lang}_lr${lr}_wu${warmup}_mu${max_update}_seed${seed}_mt${max_tokens}_patience${patience}

output_dir=/home/sata/kly/fairseq_mmt/output/textonly_baseline/${name}







cp ${BASH_SOURCE[0]} $output_dir/train.sh

gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`

cmd="fairseq-train data-bin/$data_dir
  --save-dir $output_dir
  --distributed-world-size $gpu_num -s $src_lang -t $tgt_lang
  --arch $arch
  --dropout $dropout
  --criterion $criterion --label-smoothing 0.1
  --task translation
  --optimizer adam --adam-betas '(0.9, 0.98)'
  --lr $lr --min-lr 1e-09 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup
  --max-tokens $max_tokens --update-freq $update_freq --max-update $max_update
  --find-unused-parameters
  --share-all-embeddings
  --patience $patience
  --keep-last-epochs $keep_last_epochs"

if [ $fp16 -eq 1 ]; then
cmd=${cmd}" --fp16 "
fi


export CUDA_VISIBLE_DEVICES=$device
cmd="nohup "${cmd}" > $output_dir/train.log 2>&1 &"
eval $cmd
tail -f $output_dir/train.log
