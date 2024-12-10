#!/usr/bin/bash

# conda env is "font"


# specify which GPUs you want to use.
export CUDA_VISIBLE_DEVICES="0,1,2,3"
# 使用 IFS 分割字符串并计算元素个数
IFS=',' read -r -a array <<< "$CUDA_VISIBLE_DEVICES"
n_GPU=${#array[@]}
# 输出结果
echo "The number of GPU is: $n_GPU"

# set the training args

###########################################################
# time : 2024-10-02
# pretrained from official pretrained weight with data_aug
# image_to_text_mean_rank: 1.7139      image_to_text_median_rank: 1.0000       image_to_text_R@1: 0.5626       image_to_text_R@5: 0.9947       image_to_text_R@10: 1.0000   
# text_to_image_mean_rank: 1.8966      text_to_image_median_rank: 2.0000       text_to_image_R@1: 0.4910       text_to_image_R@5: 0.9888       text_to_image_R@10: 1.0000  
# clip_val_loss: 0.0061    epoch: 0.0000   num_samples: 17120.0000
##########################################################
# torchrun --nproc_per_node 4 -m src.open_clip_train.main \
#     --batch-size 50 \
#     --precision "amp" \
#     --workers 10 \
#     --report-to "tensorboard" \
#     --save-frequency 1 \
#     --logs "logs" \
#     --dataset-type "font_csv" \
#     --csv-separator "," \
#     --train-data "data/csv/train_fontV4_s1712.csv" \
#     --val-data "data/csv/val_fontV4_s1712.csv" \
#     --csv-img-key filepath \
#     --csv-caption-key title \
#     --warmup 5000 \
#     --lr 5e-6 \
#     --wd 0.1 \
#     --epochs 10 \
#     --model ViT-L-14 \
#     --name "CLIP_font_ViT_L14_s1712_aug" \
#     --pretrained "/home/yue/DeepLearning/VISION_TEXT/pretrained_weights/OpenAI_CLIP/ViT-L-14.pt"
#     --gather-with-grad \
#     --local-loss \
#     --grad-checkpointing \
#     --train-num-samples 20000000  --dataset-resampled

###########################################################
# time : 2024-10-08
# pretrained from tiny_data weight without data_aug
# work Done it not good without WeightedSampler
##########################################################
# torchrun --nproc_per_node 4 -m src.open_clip_train.main \
#     --batch-size 100 \
#     --precision "amp" \
#     --workers 10 \
#     --report-to "tensorboard" \
#     --save-frequency 1 \
#     --logs "logs" \
#     --dataset-type "font_csv" \
#     --csv-separator "," \
#     --train-data "data/csv/train_fontV4_s1712.csv" \
#     --val-data "data/csv/val_fontV4_s1712.csv" \
#     --csv-img-key filepath \
#     --csv-caption-key title \
#     --warmup 5000 \
#     --lr 5e-6 \
#     --wd 0.1 \
#     --epochs 10 \
#     --model ViT-L-14 \
#     --name "CLIP_font_ViT_L14_s1712" \
#     --pretrained "logs/CLIP_font_ViT_L14_tiny_aug/checkpoints/epoch_30.pt" \
#     --gather-with-grad \
#     --local-loss \
#     --grad-checkpointing 

###########################################################
# time : 2024-12-02
# pretrained from L14.pt weight without data_aug but with ROPE
# Using CustomCLIP -> CLIP_Rope class train is work
# DDP not support Dynamtic Inject
##########################################################
torchrun --nproc_per_node $n_GPU -m src.open_clip_train.main \
    --batch-size 100 \
    --precision "amp" \
    --workers 40 \
    --report-to "tensorboard" \
    --save-frequency 1 \
    --logs "logs" \
    --dataset-type "font_csv" \
    --csv-separator "," \
    --train-data "data/csv/train_fontV4_s1712.csv" \
    --val-data "data/csv/val_fontV4_s1712.csv" \
    --csv-img-key filepath \
    --csv-caption-key title \
    --warmup 5000 \
    --lr 9e-6 \
    --wd 0.1 \
    --epochs 20 \
    --model ViT-L-14 \
    --name "ViT_L14_s1712_rope" \
    --pretrained "/home/yue/DeepLearning/VISION_TEXT/pretrained_weights/Open_CLIP_Torch/FontStroke/epoch_2.pt" \
    --gather-with-grad \
    --local-loss \
    --use_rope  \
    --grad-checkpointing \
    

###########################################################
# time : 2024-12-02
# pretrained from L14.pt weight without data_aug but with ROPE
# Failed
##########################################################
# torchrun --nproc_per_node 4 -m src.open_clip_train.main \
#     --batch-size 20 \
#     --precision "amp" \
#     --worker 10 \
#     --report-to "tensorboard" \
#     --save-frequency 1 \
#     --logs "logs" \
#     --dataset-type "font_csv" \
#     --csv-separator "," \
#     --train-data "data/csv/train_fontV4_s1712.csv" \
#     --val-data "data/csv/val_fontV4_s1712.csv" \
#     --csv-img-key filepath \
#     --csv-caption-key title \
#     --warmup 5000 \
#     --lr 5e-6 \
#     --wd 0.1 \
#     --epochs 20 \
#     --model ViT-H-14 \
#     --name "ViT_H14_s1712_rope" \
#     --pretrained "/home/yue/DeepLearning/VISION_TEXT/pretrained_weights/Open_CLIP_Torch/CLIP-ViT-H-14-laion2B-s32B-b79K/ViT-H-14.safetensors" \
#     --gather-with-grad \
#     --local-loss \
#     --grad-checkpointing \
#     --use_rope
