gpu_id=1
bz=32
lr=0.01

# 定义学习率列表
learning_rates=("5e-2" "1e-2" "5e-3" "1e-3" "5e-4")

# 循环遍历每个学习率
for lr in "${learning_rates[@]}"; do
    # 输出当前学习率，便于调试
    echo "Training with learning rate: $lr"
    
    # 运行训练脚本，传递当前的学习率
    python train_gps.py /home1/yhr/GPS1/data/vtab/vtab-1k/oxford_flowers102 \
        --dataset flowers102 \
        --num-classes 102 --direct-resize --no-aug \
        --model vit_base_patch16_224_in21k \
        --epochs 100 \
        --batch-size $bz \
        --opt adam --weight-decay 0 \
        --warmup-lr 1e-3 --warmup-epochs 20 \
        --lr $lr --min-lr 1e-8 \
        --drop-path 0 --img-size 224 \
        --mixup 0 --cutmix 0 --smoothing 0 \
        --output output/${lr}_output/ \
        --amp --tuning-mode part --pretrained \
        --pruning --pruning_method gradient_perCell \
        --times_para 15 \
        --gpu_id $gpu_id \
        --log-wandb \
        --experiment vtab \
        --no-prefetcher
done

echo "Training completed for all learning rates."

# python train_gps.py /home1/yhr/GPS1/data/vtab/vtab-1k/oxford_flowers102 \
#     --dataset oxford_flowers102 \
#     --num-classes 102 --direct-resize --no-aug \
#     --model vit_base_patch16_224_in21k \
#     --epochs 100 \
#     --batch-size $bz \
#     --opt adam  --weight-decay 0 \
#     --warmup-lr 1e-3 --warmup-epochs 20 \
#     --lr $lr --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
# 	  --mixup 0 --cutmix 0 --smoothing 0 \
#     --output output/ \
#     --amp --tuning-mode part --pretrained \
#     --pruning --pruning_method gradient_perCell \
#     --times_para 15 \
#     --gpu_id $gpu_id \
#     --log-wandb \
#     --experiment vtab \
#     --no-prefetcher 

# python train_gps.py path/to/vtab-1k/oxford_flowers102 \
#     --dataset flowers102 \
#     --num-classes 102 --direct-resize --no-aug \
#     --model vit_base_patch16_224_in21k \
#     --epochs 100 \
#     --batch-size $bz \
#     --opt adam  --weight-decay 0 \
#     --warmup-lr 1e-7 --warmup-epochs 10 \
#     --lr $lr --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
# 	  --mixup 0 --cutmix 0 --smoothing 0 \
#     --output /gpfs/home6/zzhang3/gps_git/output/ \
#     --amp --tuning-mode part --pretrained \
#     --pruning --pruning_method gradient_perCell \
#     --times_para 1 \
#     --gpu_id $gpu_id \
#     --log-wandb \
#     --experiment vtab \
#     --contrast-aug --no-prefetcher --contrastiveve
