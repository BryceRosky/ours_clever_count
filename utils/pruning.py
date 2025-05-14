import numpy as np
import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
import os
import re
from collections import defaultdict

def prune_by_percentile_gradient_perCell(model, time_para=1, block_ratios=None,data_set=str):
    statistic = {}
    new_masks = {}    
    # grad_tensors = {}
    import logging
    # logging.basicConfig(filename='/home1/yhr/GPS1/'+data_set+'_weight_analysis.log',  # 日志文件名
    #                     level=logging.INFO,     # 日志级别
    #                     format='%(asctime)s - %(message)s')  # 日志格式
    # print('/home1/yhr/GPS1/'+data_set+'_weight_analysis.log')
    for name, param in model.named_parameters():
        if "norm" in name or "pos_embed" in name or "cls_token" in name:
            new_mask = np.ones_like(param.data.cpu().numpy())
        elif 'head' in name or "bias" in name or "gamma" in name:
            new_mask = np.zeros_like(param.data.cpu().numpy())
        # if "norm" in name or "cls_token" in name:
        #     new_mask = np.ones_like(param.data.cpu().numpy())
        # elif "pos_embed" in name or 'head' in name or "bias" in name or "gamma" in name:
        #     new_mask = np.zeros_like(param.data.cpu().numpy())
        else:
            if "patch_embed" in name or "conv" in name or "stem.proj.weight" in name or "downsample.proj.weight" in name:
                tensor = param.grad.data.cpu().numpy()
                B,C,H,W = tensor.shape
                tensor = np.reshape(tensor,[B,-1])
            else:
                tensor = param.grad.data.cpu().numpy()
            # if "attn" in name:
            #     import pdb
            #     pdb.set_trace()
            # grad_tensors[name] = tensor.copy()
            ##########################################################################
            
            # 获取 block id
            block_id, _ = extract_block_and_submodule_from_name(name)
            adjusted_time_para = time_para

            # 如果提供了 block_ratios，则调整 time_para
            if block_ratios is not None and block_id is not None:
                adjusted_time_para *= block_ratios.get(block_id, 1)  # 若 block_id 不在 block_ratios 中，则默认为 1
            # adjusted_time_para +=1
            new_mask=np.ones_like(tensor)
            # adjusted_time_para  = [x + 1 for x in adjusted_time_para]
            # print(adjusted_time_para)
            # for ind in range(time_para):
            for ind in range(int(adjusted_time_para)):
                # max_index = abs(tensor).argsort(1)[:, -(ind + 1)] #get sort
                max_index = (tensor ** 2).argsort(1)[:, -(ind + 1)] #get sort
                one_hot_temp = ~np.eye(max(tensor.shape))[max_index][:, :tensor.shape[1]].astype(bool)
                new_mask_temp = one_hot_temp.astype(np.float32)
                new_mask = new_mask.astype(int) & new_mask_temp.astype(int)
                new_mask = new_mask.astype(np.float32)


            if "patch_embed" in name or "conv" in name or "stem.proj.weight" in name or "downsample.proj.weight" in name:
                new_mask = np.reshape(new_mask, (B,C,H,W))

        trainable_param = len(new_mask.reshape(-1))-len(np.nonzero(new_mask)[0])
        total_para = len(new_mask.reshape(-1))
        statistic[name]=[trainable_param, total_para]
        logging.info("%s: %d/%d (%.4f%%) %s", name, trainable_param, total_para, (trainable_param / total_para) * 100, new_mask.shape)
        print(name, ": ", trainable_param, "/", total_para, "(",np.round((trainable_param/total_para)*100, 4), "%)", new_mask.shape   )


        new_masks[name] = torch.from_numpy(new_mask).cuda()    

    print("---------------------------------------------------------------")
    trainable_withouthead = 0
    total_withouthead = 0
    trainable_head = 0
    total_head = 0
    for na, [trainable_p, t_p] in statistic.items():
        if "head" not in na:
            trainable_withouthead = trainable_withouthead + trainable_p
            total_withouthead = total_withouthead + t_p
        else:
            trainable_head = trainable_head + trainable_p
            total_head = total_head + t_p
    print("---------------------------------------------------------------")

    print("---------------------------------------------------------------")
    print("Trainable parameter / Total (without head): ", trainable_withouthead, "/", total_withouthead, "(", np.round((trainable_withouthead/total_withouthead)*100,4), "%)")
    print("Trainable parameter / Total (head): ", trainable_head, "/", total_head, "(", np.round((trainable_head/total_head)*100,4), "%)")
    print("Trainable parameter / Total (total): ", trainable_head+trainable_withouthead, "/", total_head+total_withouthead, "(", np.round(((trainable_head+trainable_withouthead)/(total_head+total_withouthead))*100,4), "%)")

    print("#######################################################################")
    return new_masks

def analyze_scl(grad_tensors, top_percent=0.01):
    # scl_values = []
    # scl_metadata = []
    fisher_values = []
    fisher_metadata = []
    ##############################################################################
    # block_counts = defaultdict(int)
    # total_count = 0

    for name, tensor in grad_tensors.items():
        #########################################################################
        # tensor_scl = abs(tensor).flatten()
        tensor_fisher = (tensor ** 2).flatten

        # block_id, submodule = extract_block_and_submodule_from_name(name)
        
        ##########################################################################
        # 统计每个block的参数数量
        # block_counts[block_id] += len(tensor_fisher)
        # total_count += len(tensor_fisher)

        for i, fisher_value in enumerate(tensor_fisher):
            fisher_values.append(fisher_value)
            fisher_metadata.append({
                'name': name,
                'index': i,
                'fisher': fisher_value,
                'block_id': block_id,
                'submodule': submodule,
                'param_type': 'bias' if 'bias' in name else 'weight',
            })

    fisher_values = np.array(fisher_values)
    threshold = np.percentile(fisher_values, 100 * (1 - top_percent))
    top_indices = np.where(fisher_values >= threshold)[0]

    top_params = [fisher_metadata[i] for i in top_indices]
    top_params = sorted(top_params, key=lambda x: x['fisher'], reverse=True)

    print("\n=== Top 1% Parameters fisher Analysis ===")
    print("Top 1% Params Count:", len(top_params))

    # 可以按照block和submodule分类统计一下
    block_submodule_count = {}
    for p in top_params:
        key = (p['block_id'], p['submodule'])
        if key not in block_submodule_count:
            block_submodule_count[key] = 0
        block_submodule_count[key] += 1

    # 打印分布统计信息
    print("\n=== Top 1% Parameters Distribution by Block and Submodule ===")
    
    filtered_block_submodule_count = {
    (block_id, submodule): count
    for (block_id, submodule), count in block_submodule_count.items()
    if block_id is not None
    }
    
    for (block_id, submodule), count in sorted(filtered_block_submodule_count.items()):
        if block_id is None:
            continue
        print(f"Block {block_id}, Submodule {submodule}: {count} params")

    return top_params

#############################################################################################################
def analyze_fisher(fisher_info, top_param=0.01):
    """
    计算各个 block 在 top_percent Fisher 信息中的参数比例，并进行归一化
    :param fisher_info: Fisher 信息字典 {param_name: fisher_tensor}
    :param top_percent: 选择前 top_percent 的 Fisher 信息参数
    :return: 归一化的 block 比例 {block_id: ratio}
    """
    fisher_values = []
    fisher_metadata = {}

    # 遍历所有参数，计算 Fisher 信息
    for name, tensor in fisher_info.items():
        tensor_fisher = tensor.flatten()  # 展平 Fisher 信息
        block_id, _ = extract_block_and_submodule_from_name(name)

        if block_id is not None:
            if block_id not in fisher_metadata:
                fisher_metadata[block_id] = []
            fisher_metadata[block_id].extend(tensor_fisher.tolist())

    # 计算 Fisher 信息的全局阈值
    all_fisher_values = np.concatenate([np.array(values) for values in fisher_metadata.values()])
    threshold = np.percentile(all_fisher_values, 100 * (1 - top_param))

    # 统计每个 block 中超过阈值的参数数量
    block_counts = {block_id: sum(np.array(values) >= threshold) for block_id, values in fisher_metadata.items()}

    # 归一化：最小值为 1
    min_count = min(block_counts.values()) if block_counts else 1
    block_ratios = {block_id: count / min_count for block_id, count in block_counts.items()}

    print("\n=== Top 1% Parameters Distribution by Block ===")
    for block_id, count in sorted(block_counts.items()):
        print(f"Block {block_id}: {count} params (Ratio: {block_ratios[block_id]:.2f})")

    return block_ratios

def extract_block_and_submodule(name):
    """改进的层信息解析函数"""
    # 匹配ViT风格的命名 (e.g., blocks.0.attn.qkv.weight)
    block_match = re.search(r'blocks\.(\d+)\.([a-zA-Z]+)', name)
    if block_match:
        return (int(block_match.group(1)), block_match.group(2))
    
    # 匹配CNN风格的命名 (e.g., layer1.0.conv1.weight)
    layer_match = re.search(r'layer(\d+)\.\d+\.([a-zA-Z]+)', name)
    if layer_match:
        return (int(layer_match.group(1)), layer_match.group(2))
    
    # 匹配特殊模块
    if 'patch_embed' in name:
        return (-1, 'patch_embed')
    if 'head' in name:
        return (-2, 'head')
    
    return (-3, 'others')


def extract_block_and_submodule_from_name(param_name):

    # 匹配 "blocks.<数字>.<模块名>" 结构
    block_match = re.search(r'blocks\.(\d+)\.([\w_]+)', param_name)

    if block_match:
        block_id = int(block_match.group(1))  # 提取 block 编号
        submodule = block_match.group(2)      # 提取模块名（如 norm1, attn, mlp 等）

        # 进一步处理 `attn.qkv` 和 `mlp.fc1` 这种二级嵌套情况
        if "attn" in submodule:
            return block_id, "attn"
        elif "mlp" in submodule:
            return block_id, "mlp"
        elif "norm1" in submodule:
            return block_id, "norm1"
        elif "norm2" in submodule:
            return block_id, "norm2"

        return block_id, submodule  # 其他情况直接返回

    return None, None



def extract_block_from_name(name):
    # 假设参数名字里有 block 编号，比如 blocks.3.attn.qkv.weight 这样的形式
    # 根据你的模型结构调整这里的逻辑
    if 'blocks' in name:
        parts = name.split('.')
        for i, part in enumerate(parts):
            if part == 'blocks':
                return f"block_{parts[i + 1]}"
    return 'others'


def extract_layer_from_name(name):
    # 简单提取层的信息，比如 attn、mlp 等，具体视你的模型结构而定
    if 'attn' in name:
        return 'attn'
    elif 'mlp' in name:
        return 'mlp'
    elif 'patch_embed' in name:
        return 'patch_embed'
    elif 'norm' in name:
        return 'norm'
    return 'others'


# def prune_by_fisher_information(model, time_para=1):
#     """ 使用 Fisher Information 选择可训练参数 """
#     fisher_info = {}

#     for name, param in model.named_parameters():
#         if param.grad is not None:
#             fisher_score = (param.grad ** 2).mean().item()  # 计算 Fisher 信息
#             fisher_info[name] = fisher_score
    
#     # 设定阈值，选出 Top-k 重要参数
#     threshold = np.percentile(list(fisher_info.values()), 100 - time_para)  
#     new_masks = {}
    
#     for name, param in model.named_parameters():
#         if name in fisher_info and fisher_info[name] >= threshold:
#             new_masks[name] = torch.ones_like(param, device=param.device)
#         else:
#             new_masks[name] = torch.zeros_like(param, device=param.device)

#     return new_masks
