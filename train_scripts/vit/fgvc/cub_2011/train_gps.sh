gpu_id=0
bz=32
lr=0.003
# learning_rates=("6e-4")
learning_rates=( "8e-5" "1e-4" "5e-5")

for lr in "${learning_rates[@]}"; do
python train_gps.py /home/yhr/GPS1/data/FGVC \
    --dataset cub \
    --num-classes 200 --simple-aug \
    --model vit_base_patch16_224_in21k \
    --epochs 100 \
    --batch-size $bz \
    --opt adam --weight-decay 0 \
    --warmup-lr 1e-3 --warmup-epochs 10 \
    --lr $lr --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
    --model-ema --model-ema-decay 0.9998 \
    --output drop_FGVC_output/cub_warmup1e-3_${lr}/ \
    --amp --tuning-mode part --pretrained \
    --pruning --pruning_method gradient_perCell \
    --times_para 2 \
    --gpu_id $gpu_id \
    --log-wandb \
    --experiment fgvc \
    --run_name "" \
    --no-prefetcher --keep_rate 1 --top_param 0.03 --head_reparam
done


