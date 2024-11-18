#!/usr/bin/bash

python src/open_clip_train/main.py --batch-size 50 \
    --precision "amp" \
    --worker 10 \
    --report-to "tensorboard" \
    --save-frequency 1 \
    --logs "logs" \
    --dataset-type "font_csv" \
    --csv-separator "," \
    --train-data "data/csv/train_fontV4_tiny.csv" \
    --val-data "data/csv/val_fontV4_tiny.csv" \
    --csv-img-key filepath \
    --csv-caption-key title \
    --warmup 1000 \
    --lr 5e-6 \
    --wd 0.1 \
    --epochs 30 \
    --model ViT-L-14 \
    --name "CLIP_font_ViT_L14_tiny_aug" \
    --pretrained "/home/yue/DeepLearning/VISION_TEXT/pretrained_weights/OpenAI_CLIP/ViT-L-14.pt"
