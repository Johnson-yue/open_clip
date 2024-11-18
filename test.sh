#!/usr/bin/bash

python -m src.open_clip_train.main \
    --val-data="data/csv/val_fontV4_s1712.csv" \
    --model ViT-L-14 \
    --dataset-type "font_csv" \
    --csv-separator "," \
    --csv-img-key filepath \
    --csv-caption-key title \
    --pretrained "logs/CLIP_font_ViT_L14_s1712_aug/checkpoints/epoch_8.pt"