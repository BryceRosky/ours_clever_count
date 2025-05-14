gpu_id=4
bz=16
lr=0.0003
# 定义学习率列表
learning_rates=("3e-4" "e-4" "1e-4" "4e-4")

# 定义 warmup 学习率列表
warmup_lrs=("1e-7" "2e-3" "4e-3"  )  # 根据需要增加或修改

# 设置 batch-size 和 gpu_id（确保这些变量已定义，或者直接在脚本中定义
# 外层循环遍历每个学习率
for lr in "${learning_rates[@]}"; do
    # 输出当前学习率，便于调试
    echo "Training with learning rate: $lr"
    
    # 内层循环遍历每个 warmup 学习率
    for warmup_lr in "${warmup_lrs[@]}"; do
        # 输出当前 warmup 学习率，便于调试
        echo "Training with warmup learning rate: $warmup_lr"
        
        # 运行训练脚本，传递当前的学习率和 warmup 学习率
        python train_gps.py /home1/yhr/GPS1/data/vtab/vtab-1k/smallnorb_ele \
            --dataset smallnorb_ele \
            --num-classes 9 --direct-resize --no-aug \
            --model vit_base_patch16_224_in21k \
            --epochs 100 \
            --batch-size $bz --validation-batch-size 128 \
            --opt adam --weight-decay 0 \
            --warmup-lr $warmup_lr --warmup-epochs 10 \
            --lr $lr --min-lr 1e-8 \
            --drop-path 0.1 --img-size 224 \
            --mixup 0 --cutmix 0 --smoothing 0 \
            --output please_ele_output/sep_wue20_0.01_4_0.1_drop_head_${lr}_${warmup_lr}_output/ \
            --amp --tuning-mode part --pretrained \
            --pruning --pruning_method gradient_perCell \
            --times_para 15 \
            --gpu_id $gpu_id \
            --log-wandb \
            --experiment vtab \
            --no-prefetcher --keep_rate 1 --top_param 0.01  --head_reparam
    done
done

echo "Training completed for all learning rates and warmup learning rates."
# python train_gps.py /home1/yhr/GPS1/data/vtab/vtab-1k/smallnorb_ele \
#     --dataset smallnorb_ele \
#     --num-classes 9 --direct-resize --no-aug \
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
#     --times_para 15 \
#     --gpu_id $gpu_id \
#     --log-wandb \
#     --experiment vtab \
#     --no-prefetcher 

# python train_gps.py path/to/vtab-1k/smallnorb_ele \
#     --dataset smallnorb_ele \
#     --num-classes 9 --direct-resize --no-aug \
#     --model vit_base_patch16_224_in21k \
#     --epochs 100 \
#     --batch-size $bz --validation-batch-size 128 \
#     --opt adam  --weight-decay 0 \
#     --warmup-lr 1e-7 --warmup-epochs 10 \
#     --lr $lr --min-lr 1e-8 \
#     --drop-path 0.2 --img-size 224 \
# 	--mixup 0 --cutmix 0 --smoothing 0 \
#     --output output/ \
#     --amp --tuning-mode part --pretrained \
#     --pruning --pruning_method gradient_perCell \
#     --times_para 1 \
#     --gpu_id $gpu_id \
#     --log-wandb \
#     --experiment vtab \
#     --contrast-aug --no-prefetcher --contrastive

