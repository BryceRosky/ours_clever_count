gpu_id=0
bz=32
lr=0.0005

# 定义学习率列表
# 定义学习率和 warmup-lr 的范围
# learning_rates=("1e-4" "5e-5" "2e-5" "1e-5"   "5e-4" "6e-6" "4e-6" "7e-6" "1e-3" "5e-3" )
# warmup_lrs=("3e-3" "4e-3" "1e-3" "2e-3"  "2e-4")  # 你可以根据需求设置 warmup-lr 的值
# learning_rates=("0.0001" )
# warmup_lrs=("0.003")  # 你可以根据需求设置 warmup-lr 的值

lr=0.0001
warmup_lr=0.0001
# learning_rates=("6e-4" "5e-5" "6e-5" "1e-4")
# # 定义 warmup 学习率列表
# warmup_lrs=("5e-4" "1e-3"  "2e-3" "4e-3")  # 根据需求可以修改或增加更多 warmup 学习率
keep_rates=("0.8" "0.5")
# keep_rates=("0.95" "0.9" "0.8" "0.5")
for keep_rate in "${keep_rates[@]}"; do
    echo "Training with token: $keep_rate"
    python train_gps.py /home/yhr/GPS1/data/vtab/vtab-1k/oxford_iiit_pet \
        --dataset pets \
        --num-classes 37 --direct-resize --no-aug \
        --model vit_base_patch16_224_in21k \
        --epochs 100 \
        --batch-size $bz \
        --opt adam --weight-decay 0 \
        --warmup-lr $warmup_lr --warmup-epochs 4 \
        --lr $lr --min-lr 1e-8 \
        --drop-path 0.1 --img-size 224 \
        --mixup 0 --cutmix 0 --smoothing 0 \
       --output 369_output/${keep_rate}_pets_output/ \
        --amp --tuning-mode part --pretrained \
        --pruning --pruning_method gradient_perCell \
        --times_para 15 \
        --gpu_id $gpu_id \
        --log-wandb \
        --experiment vtab \
        --no-prefetcher --keep_rate $keep_rate --top_param 0.03  --head_reparam --init_lr
done



# learning_rates=("0.0001" )
# warmup_lrs=("0.0001") 

# # 外部循环遍历每个 warmup-lr
# for warmup_lr in "${warmup_lrs[@]}"; do
#     # 输出当前 warmup-lr，便于调试
#     echo "Training with warmup-lr: $warmup_lr"
    
#     # 内部循环遍历每个学习率
#     for lr in "${learning_rates[@]}"; do
#         # 输出当前学习率，便于调试
#         echo "Training with learning rate: $lr"
        
#         # 运行训练脚本，传递当前的学习率和 warmup-lr
#         python train_gps.py /home1/yhr/GPS1/data/vtab/vtab-1k/oxford_iiit_pet \
#             --dataset pets \
#             --num-classes 37 --direct-resize --no-aug \
#             --model vit_base_patch16_224_in21k \
#             --epochs 100 \
#             --batch-size $bz \
#             --opt adam --weight-decay 0 \
#             --warmup-lr $warmup_lr --warmup-epochs 4 \
#             --lr $lr --min-lr 1e-8 \
#             --drop-path 0.1 --img-size 224 \
#             --mixup 0 --cutmix 0 --smoothing 0 \
#             --output best_output/0.95_pets_drop_${lr}_${warmup_lr}_output/ \
#             --amp --tuning-mode part --pretrained \
#             --pruning --pruning_method gradient_perCell \
#             --times_para 15 \
#             --gpu_id $gpu_id \
#             --log-wandb \
#             --experiment vtab \
#             --no-prefetcher --keep_rate 0.95 --top_param 0.03  --head_reparam
#     done
# done

# echo "Training completed for all learning rates."

# python train_gps.py /home1/yhr/GPS1/data/vtab/vtab-1k/oxford_iiit_pet \
#     --dataset oxford_iiit_pet \
#     --num-classes 37 --direct-resize --no-aug \
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

# python train_gps.py path/to/vtab-1k/oxford_iiit_pet \
#     --dataset pets \
#     --num-classes 37 --direct-resize --no-aug \
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
#     --contrast-aug --no-prefetcher --contrastive


