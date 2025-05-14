gpu_id=4
bz=32
lr=0.005

# 定义学习率列表
learning_rates=("1e-6" "5e-6" "6e-6" "5e-6" "6e-6" )

# 定义 warmup 学习率列表
warmup_lrs=("1e-4" "3e-4" "2e-4" "4e-4")   # 根据需要增加或修改


# 外层循环遍历每个学习率
for lr in "${learning_rates[@]}"; do
    # 输出当前学习率，便于调试
    echo "Training with learning rate: $lr"
    
    # 内层循环遍历每个 warmup 学习率
    for warmup_lr in "${warmup_lrs[@]}"; do
        echo "Training with learning rate: $lr"
        echo "svhn"
        # 输出当前 warmup 学习率，便于调试
        echo "Training with warmup learning rate: $warmup_lr"
        
        python train_gps.py /home1/yhr/GPS1/data/vtab/vtab-1k/svhn \
            --dataset svhn \
            --num-classes 10 --direct-resize --no-aug \
            --model vit_base_patch16_224_in21k \
            --epochs 100 \
            --batch-size $bz --validation-batch-size 128 \
            --opt adam --weight-decay 0 \
            --warmup-lr $warmup_lr --warmup-epochs 20 \
            --lr $lr --min-lr 1e-8 \
            --drop-path 0 --img-size 224 \
            --mixup 0 --cutmix 0 --smoothing 0 \
            --output small_svhn_output/${lr}_${warmup_lr}_output/ \
            --amp --tuning-mode part --pretrained \
            --pruning --pruning_method gradient_perCell \
            --times_para 15 \
            --gpu_id $gpu_id \
            --log-wandb \
            --experiment vtab \
            --no-prefetcher
    done
done

echo "Training completed for all learning rates and warmup learning rates on both datasets."
# python train_gps.py path/to/vtab-1k/svhn \
#     --dataset svhn \
#     --num-classes 10 --direct-resize --no-aug \
#     --model vit_base_patch16_224_in21k \
#     --epochs 100 \
#     --batch-size $bz \
#     --opt adam  --weight-decay 0 \
#     --warmup-lr 1e-7 --warmup-epochs 10 \
#     --lr $lr --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
# 	  --mixup 0 --cutmix 0 --smoothing 0 \
#     --output output/ \
#     --amp --tuning-mode part --pretrained \
#     --pruning --pruning_method gradient_perCell \
#     --times_para 1 \
#     --gpu_id $gpu_id \
#     --log-wandb \
#     --experiment vtab \
#     --contrast-aug --no-prefetcher --contrastive

