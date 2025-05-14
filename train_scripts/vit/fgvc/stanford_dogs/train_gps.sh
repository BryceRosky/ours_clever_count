#!/bin/bash


#Set job requirements
#SBATCH -n 1
#SBATCH -t 20:00:00
#SBATCH -p gpu
#SBATCH --gpus-per-node=1



# source activate prompt

gpu_id=2
bz=64
lr=0.0001
# learning_rates=("2e-4" "1e-4" "3e-4" "5e-4" "5e-5")
# learning_rates=("1e-4" "15e-5" "2e-4")

learning_rates=("1e-5"  "2e-5" "1e-3" "2e-3" "5e-5" "5e-3")
for lr in "${learning_rates[@]}"; do
        python train_gps.py /home/yhr/GPS1/data/FGVC \
                --dataset dogs \
                --num-classes 120 --simple-aug\
                --model vit_base_patch16_224_in21k \
                --epochs 100 \
                --batch-size $bz \
                --opt adam  --weight-decay 0 \
                --warmup-lr 1e-3 --warmup-epochs 10 \
                --lr $lr --min-lr 1e-8 \
                --drop-path 0.1 --img-size 224 \
                --model-ema --model-ema-decay 0.9998 \
                --output drop_FGVC_output/dogs_warmup1e-3_${lr}/ \
                --amp --tuning-mode part --pretrained \
                --pruning --pruning_method gradient_perCell \
                --times_para 1 \
                --gpu_id $gpu_id \
                --log-wandb \
                --experiment fgvc \
                --run_name "" \
                --no-prefetcher \
                --no-prefetcher --keep_rate 1 --top_param 0.03 --head_reparam
done
