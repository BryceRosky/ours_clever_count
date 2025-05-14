gpu_id=3
bz=16
lr=0.002

# learning_rates=( "6e-4" "4e-4" "3e-4" "7e-4" )
# learning_rates=("5e-4")
learning_rates=("5e-4")
# 循环遍历每个学习率
for lr in "${learning_rates[@]}"; do
    # 输出当前学习率，便于调试
    echo "Training with learning rate: $lr"

    # 运行训练脚本，传递当前的学习率
    python train_gps.py /home/yhr/GPS2/data/vtab/vtab-1k/clevr_count \
        --dataset clevr_count \
        --num-classes 8 --direct-resize --no-aug \
        --model vit_huge_patch14_224_in21k \
        --epochs 100 \
        --batch-size $bz \
        --opt adam --weight-decay 0 \
        --warmup-lr 5e-4 --warmup-epochs 20 \
        --lr $lr --min-lr 1e-8 \
        --drop-path 0.1 --img-size 224 \
        --mixup 0 --cutmix 0 --smoothing 0 \
        --output best_output/0.95_count_${lr}_drop_5e-4_output/ \
        --amp --tuning-mode part --pretrained \
        --pruning --pruning_method gradient_perCell \
        --times_para 15 \
        --gpu_id $gpu_id \
        --log-wandb \
        --experiment vtab \
        --no-prefetcher --keep_rate 0.95 --top_param 0.03  --head_reparam
    
    # 记录训练结果，例如保存日志文件
    echo "Finished training with learning rate: $lr" >> training_results.log
done

# python train_gps.py path/to/vtab-1k/clevr_count \
#     --dataset clevr_count \
#     --num-classes 8 --direct-resize --no-aug \
#     --model vit_base_patch16_224_in21k \
#     --epochs 100 \
#     --batch-size $bz --validation-batch-size 128 \
#     --opt adam  --weight-decay 0 \
#     --warmup-lr 1e-7 --warmup-epochs 10 \
#     --lr $lr --min-lr 1e-8 \
#     --drop-path 0.1 --img-size 224 \
# 	--mixup 0 --cutmix 0 --smoothing 0 \
#     --output output/ \
#     --amp --tuning-mode part --pretrained \
#     --pruning --pruning_method gradient_perCell \
#     --times_para 1 \
#     --gpu_id $gpu_id \
#     --log-wandb \
#     --experiment vtab \
#     --contrast-aug --no-prefetcher --contrastive

