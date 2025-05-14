gpu_id=0
bz=16
lr=0.001

# 定义学习率列表
# learning_rates=("8e-3" "9e-3" "1e-2")
# learning_rates=("1e-2" "15e-3" "1e-3" "25e-3" "8e-3" "9e-3" )
learning_rates=("8e-3" "5e-3" "1e-3")

# 循环遍历每个学习率
for lr in "${learning_rates[@]}"; do
    # 输出当前学习率，便于调试
    echo "Training with learning rate: $lr"

    # 运行训练脚本，传递当前的学习率
    python train_gps.py /home1/yhr/GPS2/data/vtab/vtab-1k/caltech101 \
        --dataset caltech101 \
        --num-classes 102 --direct-resize --no-aug \
        --model vit_base_patch16_224_in21k \
        --epochs 100 \
        --batch-size $bz \
        --opt adam  --weight-decay 0 \
        --warmup-lr 1e-7 --warmup-epochs 10 \
        --lr $lr --min-lr 1e-8 \
        --drop-path 0 --img-size 224 \
        --mixup 0 --cutmix 0 --smoothing 0 \
        --output output/warm1e-7_0.01_${lr}_head_output/ \
        --amp --tuning-mode part --pretrained \
        --pruning --pruning_method gradient_perCell \
        --times_para 15 \
        --gpu_id $gpu_id \
        --log-wandb \
        --experiment vtab \
        --no-prefetcher --top_param 0.01  --head_reparam
    
    # 记录训练结果，例如保存日志文件
    echo "Finished training with learning rate: $lr" >> training_results.log
done

# python train_gps.py path/to/vtab-1k/caltech101 \
#     --dataset caltech101 \
#     --num-classes 102 --direct-resize --no-aug \
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
