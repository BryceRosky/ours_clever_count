gpu_id=4
bz=32
lr=0.001

python train_gps.py  /home1/yhr/GPS/data/FGVC \
    --dataset flowers \
    --num-classes 102 --simple-aug \
    --model vit_base_patch16_224_in21k \
    --epochs 100 \
    --batch-size $bz \
    --opt adam  --weight-decay 0.0 \
    --warmup-lr 1e-7 --warmup-epochs 10 \
    --lr $lr --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
    --model-ema --model-ema-decay 0.99 \
    --output FGVC_output/ \
    --amp --tuning-mode part --pretrained \
    --pruning --pruning_method gradient_perCell \
    --times_para 1 \
    --gpu_id $gpu_id \
    --log-wandb \
    --experiment fgvc \
    --run_name "" \
   --no-prefetcher --keep_rate 1 --top_param 0.03  --head_reparam \
    --mixup 0 \
