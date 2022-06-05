import torch
import os
import numpy as np
import math
from PIL import Image
import cv2
import matplotlib.pyplot as plt

###
# To visualize attention maps, we need files as follows:
# Images(get from multi30k, resized to 224x224):    	'./images/*.jpg'
# Attention maps(saved while translation):     	 	    './checkpoint/*/visualization/*map.pth'
# Src_tokens(saved while translation):        	 	    './checkpoint/*/visualization/*tokens.pth'
# Origin_src_tokens(saved while translating mask0): 	'./origin_tokens/*tokens.pth'
# Translation results(saved while translating):  	    './checkpoint/*/hypo.txt'
# Dictionary of src(saved while bpe):    	 	        './dict.en.txt'
# Image filename(get from multi30k):       	 	        './test_images.txt'
###



def selective_attention_visualization(model_path):
    root_path = os.getcwd()

    # Get the translation order from 'hypo.txt'
    translation_order_list = []
    with open(os.path.join(model_path, 'hypo.txt'), 'r', encoding='utf-8') as translation_order_file:
        for line in translation_order_file:
            translation_order_list.append(int(line.strip().split('\t')[0]))

    # Get image name list from 'test_images.txt'
    test_images_filename_list = []
    with open(os.path.join(root_path, 'test_images.txt'), 'r', encoding='utf-8') as test_images_filename_file:
        for line in test_images_filename_file:
            test_images_filename_list.append(line.strip())

    # Get the dictionary{id: word} from 'dict.en.txt'
    dic_no2word = {0: '<bos>', 1: '<pad>', 2: '<eos>', 3: '<unk>'}
    with open(os.path.join(root_path, 'dict.en.txt'), 'r', encoding='utf-8') as dict_file:
        for idx, line in enumerate(dict_file):
            dic_no2word[idx+4]  = line.strip().split()[0]

    # Attention map and src_tokens are divided into 8 batches with batch_size=128 in translation
    for batch in range(8):
        # Get attention maps, src_tokens and origin_tokens
        attn_map_path = os.path.join(model_path, 'visualization', str(batch) + 'map.pth')
        src_tokens_path = os.path.join(model_path, 'visualization', str(batch) + 'tokens.pth')
        origin_tokens_path = os.path.join(root_path, 'origin_tokens', str(batch) + 'tokens.pth')

        attn_map = torch.load(attn_map_path, map_location=torch.device('cpu'))
        src_tokens = torch.load(src_tokens_path, map_location=torch.device('cpu'))
        origin_tokens = torch.load(origin_tokens_path, map_location=torch.device('cpu'))

        for sent_num in range(attn_map.shape[0]):
            filename = test_images_filename_list[translation_order_list[batch * 128 + sent_num]]
            # Images for test
            if filename != '2321764238.jpg':
                continue
            # if filename != '327955368.jpg':
            #     continue
            print(filename)

            img = Image.open(os.path.join(root_path, "images", filename), mode='r')
            plt.figure(filename, figsize=(8, 8))

            for word_num in range(attn_map.shape[1]):
                # Get the attention map for the word
                attn = attn_map[sent_num][word_num].view(24, 24).cpu().numpy()
                # Get the word with the dictionary and src_tokens
                word = src_tokens.cpu().numpy()[sent_num][word_num]
                word = dic_no2word[word]
                origin_word = origin_tokens.cpu().numpy()[sent_num][word_num]
                origin_word = dic_no2word[origin_word]

                # Skip '<pad>' and '<eos>'
                if word == '<pad>' or word == '<eos>':
                    continue

                # Show the image
                plt.subplot(math.ceil(attn_map.shape[1] / 4), 4, word_num + 1)
                plt.title(word + '-' + origin_word, fontsize=9)
                plt.imshow(img, alpha=1)
                plt.axis('off')

                img_h, img_w = img.size[0], img.size[1]
                attn = cv2.resize(attn.astype(np.float32), (img_h, img_w))
                normed_attn = attn / attn.max()
                normed_attn = (normed_attn * 255).astype('uint8')

                # Show the visual attention map of the word
                plt.imshow(normed_attn, alpha=0.4, interpolation='nearest', cmap='jet')
                plt.axis('off')
            plt.show()


if __name__ == "__main__":
    model_path  = '../checkpoints/multi30k-en2de/vit_base_patch16_384/vit_base_patch16_384-mask4'
    selective_attention_visualization(model_path)



