gpu_id=3
bz=32
lr=0.001
# 定义学习率和 warmup-lr 的范围
# learning_rates=("1e-4" "3e-4" "15e-5" "15e-5"  "25e-5" "3e-5" "4e-5" "5e-5")
# learning_rates=("6e-4"  "65e-5" "55e-5" )

lr=7e-4
warmup_lr=15e-4
# keep_rates=("0.95" "0.9" "0.8" "0.5")
keep_rates=("0.8")
for keep_rate in "${keep_rates[@]}"; do
    echo "Training with token: $keep_rate"
    
    # 运行训练脚本，传递当前的学习率和 warmup 学习率
    python train_gps.py /home/yhr/GPS1/data/vtab/vtab-1k/dtd \
        --dataset dtd \
        --num-classes 47 --direct-resize --no-aug \
        --model vit_base_patch16_224_in21k \
        --epochs 100 \
        --batch-size $bz \
        --opt adam --weight-decay 0 \
        --warmup-lr $warmup_lr --warmup-epochs 20 \
        --lr $lr --min-lr 1e-8 \
        --drop-path 0 --img-size 224 \
        --mixup 0 --cutmix 0 --smoothing 0 \
        --output 5811_output/${keep_rate}_dtd_output/ \
        --amp --tuning-mode part --pretrained \
        --pruning --pruning_method gradient_perCell \
        --times_para 15 \
        --gpu_id $gpu_id \
        --log-wandb \
        --experiment vtab \
        --no-prefetcher --keep_rate $keep_rate --top_param 0.03  --head_reparam
done



# learning_rates=("7e-4")
# warmup_lrs=("1e-3" )   # 你可以根据需求设置 warmup-lr 的值

# # 外部循环遍历每个 warmup-lr
# for warmup_lr in "${warmup_lrs[@]}"; do
#     # 输出当前 warmup-lr，便于调试
#     echo "Training with warmup-lr: $warmup_lr"
    
#     # 内部循环遍历每个学习率
#     for lr in "${learning_rates[@]}"; do
#         # 输出当前学习率，便于调试
#         echo "Training with learning rate: $lr"
        
#         # 运行训练脚本，传递当前的学习率和 warmup-lr
        # python train_gps.py /home1/yhr/GPS1/data/vtab/vtab-1k/dtd \
        #     --dataset dtd \
        #     --num-classes 47 --direct-resize --no-aug \
        #     --model vit_base_patch16_224_in21k \
        #     --epochs 100 \
        #     --batch-size $bz \
        #     --opt adam --weight-decay 0 \
        #     --warmup-lr $warmup_lr --warmup-epochs 20 \
        #     --lr $lr --min-lr 1e-8 \
        #     --drop-path 0 --img-size 224 \
        #     --mixup 0 --cutmix 0 --smoothing 0 \
        #     --output best_output/dtd_noinit_${lr}_${warmup_lr}_output/ \
        #     --amp --tuning-mode part --pretrained \
        #     --pruning --pruning_method gradient_perCell \
        #     --times_para 15 \
        #     --gpu_id $gpu_id \
        #     --log-wandb \
        #     --experiment vtab \
        #     --no-prefetcher --keep_rate 0.95 --top_param 0.03  --head_reparam
#     done
# done

# echo "Training completed for all learning rates and warmup-lr combinations."

# python train_gps.py /home1/yhr/GPS1/data/vtab/vtab-1k/dtd \
#     --dataset dtd \
#     --num-classes 47 --direct-resize --no-aug \
#     --model vit_base_patch16_224_in21k \
#     --epochs 100 \
#     --batch-size $bz \
#     --opt adam  --weight-decay 0 \
#     --warmup-lr 1e-3 --warmup-epochs 10 \
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

# python train_gps.py path/to/vtab-1k/dtd \
#     --dataset dtd \
#     --num-classes 47 --direct-resize --no-aug \
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



