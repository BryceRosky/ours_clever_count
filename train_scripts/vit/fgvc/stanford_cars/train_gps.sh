
gpu_id=2
bz=32
lr=0.0005

# learning_rates=("5e-4" "1e-4" "1e-3" "5e-3" "1e-6" "5e-6")
# learning_rates=("1e-4" "2e-4" "3e-4" ""5e-4" 8e-4" )
learning_rates=("5e-3" )


for lr in "${learning_rates[@]}"; do
    # 输出当前学习率，便于调试
    echo "Training with learning rate: $lr"
    python train_gps.py /home/yhr/GPS1/data/FGVC \
        --dataset cars \
        --num-classes 196 --simple-aug \
        --model vit_base_patch16_224_in21k \
        --epochs 100 \
        --batch-size $bz \
        --opt adam  --weight-decay 0.0 \
        --warmup-lr 1e-4 --warmup-epochs 10 \
        --lr $lr --min-lr 1e-8 \
        --drop-path 0 --img-size 224 \
        --model-ema --model-ema-decay 0.9998 \
        --output drop_FGVC_output/cars_warmup4e-3_${lr}/ \
        --amp --tuning-mode part --pretrained \
        --pruning --pruning_method gradient_perCell \
        --times_para 15 \
        --gpu_id $gpu_id \
        --log-wandb \
        --experiment fgvc \
        --run_name "" \
        --mixup 0 \
        --no-prefetcher --keep_rate 1 --top_param 0.03 --head_reparam 
done
echo "Training completed for all learning rates and warmup learning rates."
# python train_gps.py /home1/yhr/GPS/data/FGVC \
#     --dataset cars \
#     --num-classes 196 --simple-aug \
#     --model vit_base_patch16_224_in21k \
#     --epochs 3 \
#     --batch-size $bz \
#     --opt adam  --weight-decay 0.0 \
#     --warmup-lr 1e-7 --warmup-epochs 10 \
#     --lr $lr --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
#     --model-ema --model-ema-decay 0.9998 \
#     --output output/ \
#     --amp --tuning-mode part --pretrained \
#     --pruning --pruning_method gradient_perCell \
#     --times_para 15 \
#     --gpu_id $gpu_id \
#     --log-wandb \
#     --experiment fgvc \
#     --run_name "" \
#     --no-prefetcher \
#     --contrast-aug \
#     --contrastive
    # --pruning --pruning_method fisher_information \