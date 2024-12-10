#!/usr/bin/bash

# conda env is "font"

# specify which GPUs you want to use.
export CUDA_VISIBLE_DEVICES="0,1,2,3"
# 使用 IFS 分割字符串并计算元素个数
IFS=',' read -r -a array <<< "$CUDA_VISIBLE_DEVICES"
n_GPU=${#array[@]}
# 输出结果
echo "The number of GPU is: $n_GPU"

torchrun --nproc_per_node $n_GPU -m  src.open_clip_train.main --batch-size 50 \
    --precision "amp" \
    --workers 20 \
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
    --epochs 40 \
    --model ViT-L-14 \
    --name "ViT_L14_tiny_rope" \
    --pretrained "/home/yue/DeepLearning/VISION_TEXT/pretrained_weights/OpenAI_CLIP/ViT-L-14.pt" \
    --gather-with-grad \
    --local-loss \
    --grad-checkpointing \
    --use_rope

