gpu_id=2
bz=32
lr=0.015


lr=6e-4
warmup_lr=4e-3
# keep_rates=("0.95" "0.9" "0.8" "0.5")
keep_rates=("0.95" "0.9" "0.8" "0.5")
for keep_rate in "${keep_rates[@]}"; do
    echo "Training with token: $keep_rate"

    python train_gps.py /home/yhr/GPS1/data/vtab/vtab-1k/kitti \
        --dataset kitti \
        --num-classes 4 --direct-resize --no-aug \
        --model vit_base_patch16_224_in21k \
        --epochs 100 \
        --batch-size $bz --validation-batch-size 128 \
        --opt adam --weight-decay 0 \
        --warmup-lr $warmup_lr --warmup-epochs 20 \
        --lr $lr --min-lr 1e-8 \
        --drop-path 0 --img-size 224 \
        --mixup 0 --cutmix 0 --smoothing 0 \
        --output 5811_output/${keep_rate}_kitti_output/ \
        --amp --tuning-mode part --pretrained \
        --pruning --pruning_method gradient_perCell \
        --times_para 15 \
        --gpu_id $gpu_id \
        --log-wandb \
        --experiment vtab \
        --no-prefetcher --keep_rate $keep_rate --top_param 0.03 --head_reparam --init_lr
done






# 定义多个学习率
# learning_rates=("6e-4" )  # 你可以加上其他学习率值

# # 定义 warmup-lr 列表
# # warmup_lrs=("4e-3")  # 你可以根据需求调整 warmup-lr 的值
# warmup_lrs=("4e-3") 
# #wue为4效果不好哈
# #不要用drop

# # 外层循环遍历每个学习率
# for lr in "${learning_rates[@]}"; do
#     # 输出当前学习率，便于调试
#     echo "Training with learning rate: $lr"
    
#     # 内层循环遍历每个 warmup-lr
#     for warmup_lr in "${warmup_lrs[@]}"; do
#         # 输出当前 warmup-lr，便于调试
#         echo "Training with warmup-lr: $warmup_lr"
        
#         # 运行训练脚本，传递当前的学习率和 warmup-lr
        # python train_gps.py /home1/yhr/GPS1/data/vtab/vtab-1k/kitti \
        #     --dataset kitti \
        #     --num-classes 4 --direct-resize --no-aug \
        #     --model vit_base_patch16_224_in21k \
        #     --epochs 100 \
        #     --batch-size $bz --validation-batch-size 128 \
        #     --opt adam --weight-decay 0 \
        #     --warmup-lr $warmup_lr --warmup-epochs 20 \
        #     --lr $lr --min-lr 1e-8 \
        #     --drop-path 0 --img-size 224 \
        #     --mixup 0 --cutmix 0 --smoothing 0 \
        #     --output best_output/0.95_kitty_no_init_${lr}_${warmup_lr}_output/ \
        #     --amp --tuning-mode part --pretrained \
        #     --pruning --pruning_method gradient_perCell \
        #     --times_para 15 \
        #     --gpu_id $gpu_id \
        #     --log-wandb \
        #     --experiment vtab \
        #     --no-prefetcher --keep_rate 0.95 --top_param 0.03 --head_reparam
#     done
# done

# echo "Training completed for all learning rates."


# python train_gps.py path/to/vtab-1k/kitti \
#     --dataset kitti \
#     --num-classes 4 --direct-resize --no-aug \
#     --model vit_base_patch16_224_in21k \
#     --epochs 100 \
#     --batch-size $bz --validation-batch-size 128 \
#     --opt adam  --weight-decay 0 \
#     --warmup-lr 1e-7 --warmup-epochs 10 \
#     --lr $lr --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
# 	--mixup 0 --cutmix 0 --smoothing 0 \
#     --output output/ \
#     --amp --tuning-mode part --pretrained \
#     --pruning --pruning_method gradient_perCell \
#     --times_para 1 \
#     --gpu_id $gpu_id \
#     --log-wandb \
#     --experiment vtab \
#     --contrast-aug --no-prefetcher --contrastive

