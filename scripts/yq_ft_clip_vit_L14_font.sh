#!/usr/bin/bash

cd open_clip/src

# specify which GPUs you want to use.
export CUDA_VISIBLE_DEVICES="0,1,2"

# set the training args
torchrun --nproc_per_node 3 -m open_clip_train.main \
    --batch-size 50 \
    --precision amp \
    --workers 4 \
    --report-to tensorboard \
    --save-frequency 1\
    --logs="logs" \
    --dataset-type font_csv \
    --csv-separator "," \
    --train-data "data/csv/train_fontV4_s1712.csv" \
    --val-data "data/csv/val_fontV4_s1712.csv" \
    --csv-img-key filepath \
    --csv-caption-key title \
    --warmup 5000 \
    --lr 5e-6 \
    --wd 0.1 \
    --epochs 20 \
    --model ViT-L-14 \
    --name "CLIP_font_ViT_L14_s1712" \
    --pretrained "/home/yue/DeepLearning/VISION_TEXT/pretrained_weights/OpenAI_CLIP/ViT-L-14.pt" \
    --gather-with-grad \
    --local-loss \
    --grad-checkpointing \
    --train-num-samples 15000000  --dataset-resampled  \