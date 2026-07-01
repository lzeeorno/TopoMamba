import os
import sys
import gc
# 在导入任何模块之前设置环境变量来抑制torchvision警告
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 设置PyTorch内存分配策略以减少碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

if __name__ == "__main__" and "--smoke" in sys.argv:
    from tools.smoke_entrypoints import run_smoke_from_config
    run_smoke_from_config("configs.config_setting_synapse", "test_synapse.py")


def _is_synapse3d_request(argv):
    from tools.synapse3d_registry import is_synapse3d_model

    if "-h" in argv or "--help" in argv:
        return False
    for index, item in enumerate(argv):
        if item == "--network" and index + 1 < len(argv):
            return is_synapse3d_model(argv[index + 1])
        if item.startswith("--network="):
            return is_synapse3d_model(item.split("=", 1)[1])
    config_path = os.path.join(os.path.dirname(__file__), "configs", "config_setting_synapse.py")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("network ="):
                    return is_synapse3d_model(stripped.split("=", 1)[1].strip().strip('"').strip("'"))
    except OSError:
        pass
    return False


if __name__ == "__main__" and "--smoke" not in sys.argv and _is_synapse3d_request(sys.argv[1:]):
    from tools.synapse3d_pipeline import run_synapse3d_test_from_legacy_argv

    run_synapse3d_test_from_legacy_argv(sys.argv[1:])
    sys.exit(0)

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import cv2
import json
import csv
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

from datasets.dataset import RandomGenerator
from engine_synapse import val_one_epoch_with_visualization

def _missing_optional_model(name: str, module_path: str):
    def _raiser(*args, **kwargs):
        raise ImportError(
            f"{name} is not included in the public TopoMamba Synapse release. "
            f"This release supports TopoMamba_2D_* and TopoMamba_3D_t only. "
            f"Missing optional module: {module_path}"
        )
    return _raiser


try:
    from models.vmunet.vmunet import VMUNet
except ImportError:
    VMUNet = _missing_optional_model("VMUNet", "models.vmunet.vmunet")
try:
    from models.DZZMamba import create_model as create_dzz_model
except ImportError:
    create_dzz_model = _missing_optional_model("DZZMamba", "models.DZZMamba")
try:
    from models.HBFormer import create_hbformer_model
except ImportError:
    create_hbformer_model = _missing_optional_model("HBFormer", "models.HBFormer")
try:
    from models.SMAFormer_lits import create_smaformer_lits_model
except ImportError:
    create_smaformer_lits_model = _missing_optional_model("SMAFormer", "models.SMAFormer_lits")
try:
    from models.SMAFormer_V3 import create_smaformer_v3_model
except ImportError:
    create_smaformer_v3_model = _missing_optional_model("SMAFormer_V3", "models.SMAFormer_V3")
try:
    from models.DWSegNet import create_dwsegnet_model
except ImportError:
    create_dwsegnet_model = _missing_optional_model("DWSegNet", "models.DWSegNet")
try:
    from models.AFFSegNet import create_affsegnet_model
except ImportError:
    create_affsegnet_model = _missing_optional_model("AFFSegNet", "models.AFFSegNet")
from models.TopoMamba_2D import create_model as create_topomamba2d_model

from utils import *
from configs.config_setting_synapse import setting_config
from tools.experiment_overrides import (
    add_shared_experiment_args,
    apply_shared_experiment_overrides,
    print_shared_override_report,
)
from tools.synapse3d_registry import is_synapse3d_model
from tools.synapse_visualization_utils import (
    colorize_label_mask,
    normalize_gray_image,
    overlay_label_mask,
)

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", "Failed to load image Python extension")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning)
# 忽略thop相关的INFO信息
logging.getLogger('thop').setLevel(logging.WARNING)


def save_test_results(config, results, save_dir):
    """保存测试结果到CSV和JSON文件"""
    
    # 保存详细结果到JSON
    json_file = os.path.join(save_dir, 'test_results_detailed.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 保存汇总结果到CSV
    csv_file = os.path.join(save_dir, 'test_results_summary.csv')
    
    organ_names = ['Aorta', 'Gallbladder', 'Left_Kidney', 'Right_Kidney', 'Liver', 'Pancreas', 'Spleen', 'Stomach']
    
    # 准备CSV数据
    csv_data = {
        'model': config.network,
        'test_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'avg_dice': results['summary']['avg_dice'],
        'std_dice': results['summary'].get('std_dice', 0.0),
        'mean_hd95': results['summary']['mean_hd95'],
        'std_hd95': results['summary'].get('std_hd95', 0.0),
    }
    
    # 添加每个器官的指标
    for i, organ in enumerate(organ_names):
        if i < len(results['summary']['dice_per_organ']):
            csv_data[f'dice_{organ}'] = results['summary']['dice_per_organ'][i]
        if i < len(results['summary']['hd95_per_organ']):
            csv_data[f'hd95_{organ}'] = results['summary']['hd95_per_organ'][i]
    
    # 写入CSV
    file_exists = os.path.exists(csv_file)
    with open(csv_file, 'a', newline='') as f:
        fieldnames = ['model', 'test_time', 'avg_dice', 'std_dice', 'mean_hd95', 'std_hd95'] + \
                    [f'dice_{organ}' for organ in organ_names] + \
                    [f'hd95_{organ}' for organ in organ_names]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(csv_data)
    
    print(f"✅ 测试结果已保存:")
    print(f"   📄 详细结果: {json_file}")
    print(f"   📊 汇总结果: {csv_file}")


def create_test_visualization_dirs(test_save_path):
    """创建测试可视化目录"""
    attention_vis_dir = os.path.join(test_save_path, 'attention_heatmaps')
    comparison_dir = os.path.join(test_save_path, 'comparison_images')
    
    os.makedirs(attention_vis_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    
    return attention_vis_dir, comparison_dir


def _spatial_map_from_feature(output, reduction='mean'):
    """Convert BCHW or BHWC feature tensors to a BCHW spatial response map."""
    if isinstance(output, (tuple, list)):
        output = next((item for item in output if isinstance(item, torch.Tensor)), None)
    if not isinstance(output, torch.Tensor):
        return None

    if len(output.shape) == 4:
        # TopoMamba/VSS blocks often return BHWC, while conv heads return BCHW.
        if output.shape[1] == output.shape[2] and output.shape[3] != output.shape[1]:
            output = output.permute(0, 3, 1, 2).contiguous()
        if reduction == 'var':
            return torch.var(output, dim=1, keepdim=True)
        if reduction == 'max':
            return torch.max(output, dim=1, keepdim=True)[0]
        return torch.mean(output, dim=1, keepdim=True)

    if len(output.shape) == 3:
        return output.unsqueeze(1)

    return None


def extract_attention_maps(model, input_tensor):
    """提取模型的注意力图 - 限制数量避免内存溢出
    
    Args:
        model: PyTorch模型
        input_tensor: 输入tensor
    Returns:
        List[Tensor]: 注意力/空间权重图集合（限制数量）
    """
    attention_weights = []
    
    def attention_hook_fn(module, input, output):
        # 限制提取数量避免OOM
        if len(attention_weights) >= 15:  # 最多提取15个attention map
            return
            
        # 专门提取attention权重
        if hasattr(module, 'attn') or 'Attention' in str(type(module)):
            # 对于标准的self-attention层
            if isinstance(output, tuple) and len(output) >= 2:
                # output通常是(value, attention_weights)
                attn_weights = output[1] if output[1] is not None else output[0]
                attention_map = _spatial_map_from_feature(attn_weights, reduction='mean')
                if attention_map is not None:
                    attention_weights.append(attention_map.detach().cpu())
            elif isinstance(output, torch.Tensor):
                # 如果只有输出值，计算其spatial attention
                attn = _spatial_map_from_feature(output, reduction='mean')
                if attn is not None:
                    attention_weights.append(attn.detach().cpu())
        
        # 对于Mamba/SS2D层，提取特征重要性作为attention
        elif any(x in str(type(module)) for x in ['VSSBlock', 'SS2D', 'SelectiveScan']):
            spatial_attn = _spatial_map_from_feature(output, reduction='var')
            if spatial_attn is not None:
                attention_weights.append(spatial_attn.detach().cpu())
    
    # 只注册关键层的attention钩子
    attention_hooks = []
    registered_count = 0
    module_name_keywords = ['encoder', 'decoder', 'layers', 'layers_up', 'self_attention']
    for name, module in model.named_modules():
        # 只选择encoder/decoder以及TopoMamba layers/layers_up中的关键层
        if (any(x in str(type(module)) for x in ['Attention', 'MultiHeadAttention', 'VSSBlock', 'SS2D']) and \
           any(keyword in name.lower() for keyword in module_name_keywords)):
            if registered_count < 20:  # 限制注册数量
                hook = module.register_forward_hook(attention_hook_fn)
                attention_hooks.append(hook)
                registered_count += 1
    
    # 前向传播
    with torch.no_grad():
        model_output = model(input_tensor)
    
    # 移除钩子
    for hook in attention_hooks:
        hook.remove()
    
    # 清理
    attention_hooks.clear()
    del attention_hooks

    if not attention_weights:
        fallback_attention = _spatial_map_from_feature(model_output, reduction='mean')
        if fallback_attention is not None:
            attention_weights.append(fallback_attention.detach().cpu())
    
    return attention_weights


def save_attention_heatmaps(image, attention_maps, case_name, save_dir, slice_idx=None):
    """保存注意力热图 - 深蓝色背景+红绿色注意力+颜色条，只保存一个随机选择的slice
    
    Args:
        image: 输入图像(Numpy)，2D或3D
        attention_maps: 注意力图集合
        case_name: 样本名
        save_dir: 保存目录
        slice_idx: 可选，三维体数据的切片索引
    Returns:
        List[str]: 保存的图像路径
    """
    
    if len(attention_maps) == 0:
        return []
    
    saved_paths = []
    
    # 如果是3D数据，随机选择一个slice保存
    if len(image.shape) == 3:
        import random
        num_slices = image.shape[0]
        # 随机选择一个slice，但避免选择边缘的slice（前10%和后10%）
        start_slice = max(0, int(num_slices * 0.1))
        end_slice = min(num_slices, int(num_slices * 0.9))
        selected_slice = random.randint(start_slice, end_slice-1) if end_slice > start_slice else num_slices // 2
        
        image_slice = image[selected_slice]
        
        # 归一化图像到0-1
        if image_slice.max() > 1.1:
            image_slice = image_slice / 255.0
        
        # 创建注意力热图可视化 - 显示所有层
        num_layers = len(attention_maps)
        if num_layers == 0:
            return []
            
        # 动态计算子图布局
        cols = min(6, num_layers)  # 最多6列
        rows = (num_layers + cols - 1) // cols  # 向上取整
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*2))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # 设置深蓝色背景
        fig.patch.set_facecolor('#0a1929')  # 深蓝色背景
        
        for i in range(num_layers):
            attn_map = attention_maps[i]
            
            # 处理注意力图尺寸
            if len(attn_map.shape) == 4:  # (B, C, H, W)
                attn_map = attn_map[0, 0]  # 取第一个batch和第一个通道
            elif len(attn_map.shape) == 3:  # (C, H, W)
                attn_map = attn_map[0]
            
            # 调整注意力图尺寸到与输入图像相同
            target_size = image_slice.shape
            if attn_map.shape != target_size:
                attn_map = cv2.resize(
                    attn_map.numpy(), 
                    (target_size[1], target_size[0]), 
                    interpolation=cv2.INTER_LINEAR
                )
            else:
                attn_map = attn_map.numpy()
            
            # 归一化注意力图到0-1
            attn_min, attn_max = attn_map.min(), attn_map.max()
            if attn_max > attn_min:
                attn_map = (attn_map - attn_min) / (attn_max - attn_min)
            else:
                attn_map = np.zeros_like(attn_map)
            
            # 创建深蓝色背景
            background = np.full_like(image_slice, 0.05)  # 深蓝色背景值
            
            # 显示注意力热图
            im = axes[i].imshow(background, cmap='Blues', vmin=0, vmax=1)
            attention_overlay = axes[i].imshow(attn_map, cmap='RdYlGn', alpha=0.8, vmin=0, vmax=1)
            
            # 设置标题和样式
            axes[i].set_title(f'Attention Layer {i+1}', fontsize=10, color='white', fontweight='bold')
            axes[i].axis('off')
            axes[i].set_facecolor('#0a1929')
            
            # 添加颜色条
            cbar = plt.colorbar(attention_overlay, ax=axes[i], fraction=0.046, pad=0.04)
            cbar.set_label('Attention Weight', color='white', fontsize=8)
            cbar.ax.tick_params(colors='white', labelsize=6)
        
        # 隐藏多余的子图
        total_subplots = rows * cols
        for i in range(num_layers, total_subplots):
            axes[i].axis('off')
            axes[i].set_facecolor('#0a1929')
        
        # 添加总标题
        fig.suptitle(f'Attention Heatmaps - {case_name} - Slice {selected_slice:03d} (Random)', 
                    fontsize=14, color='white', fontweight='bold', y=0.95)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(save_dir, f'{case_name}_random_slice{selected_slice:03d}_attention_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='#0a1929', edgecolor='none')
        plt.close()
        saved_paths.append(save_path)
        
        print(f"🎯 已保存注意力热图: {case_name} - 随机选择slice {selected_slice}")
    else:
        # 2D数据，保存单个图像（保持原有逻辑）
        image_slice = image
        # ... 2D处理代码可以保持不变 ...
    
    return saved_paths


def save_prediction_comparison(image, ground_truth, prediction, case_name, save_dir, slice_idx=None):
    """保存输入图像、GT mask、预测mask和预测叠加图。"""
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []

    image = np.asarray(image)
    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)

    if image.ndim == 3 and image.shape[0] > 3:
        if slice_idx is None:
            slice_indices = range(image.shape[0])
        else:
            slice_indices = [int(slice_idx)]
    else:
        slice_indices = [None]

    for current_slice_idx in slice_indices:
        if current_slice_idx is None:
            image_slice = image
            gt_slice = ground_truth
            pred_slice = prediction
            save_name = f'{case_name}_comparison.png'
            title = case_name
        else:
            image_slice = image[current_slice_idx]
            gt_slice = ground_truth[current_slice_idx]
            pred_slice = prediction[current_slice_idx]
            save_name = f'{case_name}_slice{current_slice_idx:03d}_comparison.png'
            title = f'{case_name} - Slice {current_slice_idx:03d}'

        image_gray = normalize_gray_image(image_slice)
        gt_mask_rgb = colorize_label_mask(gt_slice)
        pred_mask_rgb = colorize_label_mask(pred_slice)
        pred_overlay_rgb = overlay_label_mask(image_slice, pred_slice, alpha=0.72)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].imshow(image_gray, cmap='gray', vmin=0, vmax=255)
        axes[0, 0].set_title('Input Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(gt_mask_rgb)
        axes[0, 1].set_title('Ground Truth Mask', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(pred_mask_rgb)
        axes[1, 0].set_title('Prediction Mask', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(pred_overlay_rgb)
        axes[1, 1].set_title('Prediction Overlay', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, dpi=220, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        saved_paths.append(save_path)

    return saved_paths


def extract_activation_maps(model, input_tensor):
    """提取模型的激活图（特征响应），提取关键层避免内存溢出
    
    Args:
        model: PyTorch模型
        input_tensor: 输入tensor
    Returns:
        List[Tensor]: 激活图集合（限制数量以避免OOM）
    """
    activation_maps = []
    
    def hook_fn(module, input, output):
        # 提取关键层的激活特征
        if len(activation_maps) < 20:  # 只提取前20个关键层，避免内存溢出
            max_activation = _spatial_map_from_feature(output, reduction='max')
            if max_activation is not None:
                activation_maps.append(max_activation.detach().cpu())
    
    # 只注册关键层的钩子，减少内存占用
    hooks = []
    # 更有针对性的层选择，主要关注encoder/decoder的主要层
    target_modules = ['Conv2d', 'VSSBlock', 'SS2D']
    module_name_keywords = ['encoder', 'decoder', 'layers', 'layers_up', 'self_attention']
    
    registered_count = 0
    for name, module in model.named_modules():
        module_type = type(module).__name__
        # 只选择encoder/decoder以及TopoMamba layers/layers_up的主要卷积层和VSSBlock
        if any(target in module_type for target in target_modules) and \
           any(keyword in name.lower() for keyword in module_name_keywords):
            # 限制注册的钩子数量，避免过多
            if registered_count < 30:  # 最多注册30个钩子
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
                registered_count += 1
    
    # 前向传播
    with torch.no_grad():
        model_output = model(input_tensor)
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 清理hooks列表
    hooks.clear()
    del hooks
    
    # 如果没有提取到激活图，返回一个默认的激活图
    if not activation_maps:
        print("⚠️ 没有提取到激活图，使用模型输出作为fallback")
        activation_map = _spatial_map_from_feature(model_output, reduction='mean')
        if activation_map is not None:
            activation_maps.append(activation_map.detach().cpu())
    
    print(f"🎉 最终提取到 {len(activation_maps)} 个激活图")
    return activation_maps


def save_activation_heatmaps(image, activation_maps, case_name, save_dir, slice_idx=None):
    """保存激活热图 - 深蓝色背景+红绿色激活+颜色条，只保存指定层的随机选择slice
    
    Args:
        image: 输入图像(Numpy)，2D或3D
        activation_maps: 激活图集合
        case_name: 样本名
        save_dir: 保存目录
        slice_idx: 可选，固定切片索引
    Returns:
        List[str]: 保存的图像路径
    """
    
    # 指定要保存的层索引
    target_layers = [3, 13, 25, 35, 47, 57, 58, 67, 77, 87, 99, 109, 123, 133, 175, 185, 195, 205, 213]
    
    # 过滤出目标层的激活图
    filtered_activation_maps = []
    for i, layer_idx in enumerate(target_layers):
        if layer_idx - 1 < len(activation_maps):  # 转换为0-based索引
            filtered_activation_maps.append(activation_maps[layer_idx - 1])
    
    print(f"🔥 开始生成激活热图: {case_name}, 目标层数: {len(target_layers)}, 实际可用层数: {len(filtered_activation_maps)}, 保存目录: {save_dir}")
    
    if len(filtered_activation_maps) == 0:
        print("⚠️ 没有可用的目标层激活图")
        return []
    
    saved_paths = []
    
    # 如果是3D数据，随机选择一个slice保存
    if len(image.shape) == 3:
        import random
        num_slices = image.shape[0]
        # 随机选择一个slice，但避免选择边缘的slice（前10%和后10%）
        start_slice = max(0, int(num_slices * 0.1))
        end_slice = min(num_slices, int(num_slices * 0.9))
        selected_slice = random.randint(start_slice, end_slice-1) if end_slice > start_slice else num_slices // 2
        
        image_slice = image[selected_slice]
        
        # 归一化图像到0-1
        if image_slice.max() > 1.1:
            image_slice = image_slice / 255.0
        
        # 创建激活热图可视化 - 显示指定层
        num_layers = len(filtered_activation_maps)
        if num_layers == 0:
            return []
            
        # 动态计算子图布局
        cols = min(6, num_layers)  # 最多6列
        rows = (num_layers + cols - 1) // cols  # 向上取整
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*2))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # 设置深蓝色背景
        fig.patch.set_facecolor('#0a1929')  # 深蓝色背景
        
        for i in range(num_layers):
            activation_map = filtered_activation_maps[i]
            actual_layer_idx = target_layers[i] if i < len(target_layers) else i + 1
            
            # 处理激活图尺寸
            if len(activation_map.shape) == 4:  # (B, C, H, W)
                activation_map = activation_map[0].mean(dim=0)  # 取第一个batch并沿通道维度平均
            elif len(activation_map.shape) == 3:  # (C, H, W)
                activation_map = activation_map.mean(dim=0)  # 沿通道维度平均
            
            # 调整激活图尺寸到与输入图像相同
            target_size = image_slice.shape
            if activation_map.shape != target_size:
                activation_map = cv2.resize(
                    activation_map.numpy(), 
                    (target_size[1], target_size[0]), 
                    interpolation=cv2.INTER_LINEAR
                )
            else:
                activation_map = activation_map.numpy()
            
            # 归一化激活图到0-1
            act_min, act_max = activation_map.min(), activation_map.max()
            if act_max > act_min:
                activation_map = (activation_map - act_min) / (act_max - act_min)
            else:
                activation_map = np.zeros_like(activation_map)
            
            # 创建深蓝色背景
            background = np.full_like(image_slice, 0.05)  # 深蓝色背景值
            
            # 显示激活热图
            im = axes[i].imshow(background, cmap='Blues', vmin=0, vmax=1)
            activation_overlay = axes[i].imshow(activation_map, cmap='RdYlGn', alpha=0.8, vmin=0, vmax=1)
            
            # 设置标题和样式
            axes[i].set_title(f'Layer {actual_layer_idx}', fontsize=10, color='white', fontweight='bold')
            axes[i].axis('off')
            axes[i].set_facecolor('#0a1929')
            
            # 添加颜色条
            cbar = plt.colorbar(activation_overlay, ax=axes[i], fraction=0.046, pad=0.04)
            cbar.set_label('Activation Strength', color='white', fontsize=8)
            cbar.ax.tick_params(colors='white', labelsize=6)
        
        # 隐藏多余的子图
        total_subplots = rows * cols
        for i in range(num_layers, total_subplots):
            axes[i].axis('off')
            axes[i].set_facecolor('#0a1929')
        
        # 添加总标题
        fig.suptitle(f'Selected Activation Heatmaps - {case_name} - Slice {selected_slice:03d} (Random)', 
                    fontsize=14, color='white', fontweight='bold', y=0.95)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(save_dir, f'{case_name}_random_slice{selected_slice:03d}_selected_activation_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='#0a1929', edgecolor='none')
        plt.close()
        saved_paths.append(save_path)
        
        print(f"🎯 已保存选定层激活热图: {case_name} - 随机选择slice {selected_slice}, 包含 {num_layers} 个指定层")
    else:
        # 2D数据，保存单个图像（保持原有逻辑）
        image_slice = image
        # ... 2D处理代码可以保持不变 ...
    
    return saved_paths


def main():
    config = setting_config
    
    print('#----------创建测试环境----------#')
    print(f"🔧 当前网络架构: {config.network}")
    print(f"🔧 模型配置: {config.model_config}")
    
    # 创建测试输出目录
    test_work_dir = f'test_results/{config.network}_synapse'
    os.makedirs(test_work_dir, exist_ok=True)
    
    # 创建可视化目录
    attention_vis_dir, comparison_dir = create_test_visualization_dirs(test_work_dir)
    
    # 添加activation heatmap目录
    activation_vis_dir = os.path.join(test_work_dir, 'activation_heatmaps')
    os.makedirs(activation_vis_dir, exist_ok=True)
    
    # 创建日志
    log_dir = os.path.join(test_work_dir, 'log')
    global logger
    logger = get_logger('test', log_dir)
    
    print('#----------GPU初始化----------#')
    set_seed(config.seed)
    
    # 内存优化：清理缓存和垃圾回收
    torch.cuda.empty_cache()
    gc.collect()
    
    # 打印当前GPU内存使用情况
    if torch.cuda.is_available():
        print(f"📊 GPU内存: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")
        print(f"💾 系统内存优化: pin_memory=False, num_workers=0")
    
    print('#----------准备数据集----------#')
    val_dataset = config.datasets(base_dir=config.volume_path, split="test_vol", list_dir=config.list_dir)
    val_loader = DataLoader(val_dataset,
                           batch_size=1,
                           shuffle=False,
                           pin_memory=False,  # 关闭pin_memory以减少内存占用
                           num_workers=config.num_workers,      # 测试时使用单进程避免内存溢出
                           drop_last=True)
    
    print('#----------准备模型----------#')
    model_cfg = config.model_config
    
    if config.network == 'vmunet':
        model = VMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=None,  # 测试时不使用预训练权重
        )
        
    elif config.network.startswith('cumamba_v3'):
        model = create_dzz_model(
            model_cfg['model_name'],
            in_chans=model_cfg['input_channels'],
            num_classes=model_cfg['num_classes']
        )
    
    elif config.network.startswith('dzzmamba'):
        model = create_dzz_model(
            model_cfg['model_name'],
            in_chans=model_cfg['input_channels'],
            num_classes=model_cfg['num_classes'],
            branch_mode=model_cfg['branch_mode'],
            enable_cache=model_cfg['enable_cache'],
            enable_gating=model_cfg['enable_gating'],
            mia_gate_type=model_cfg['mia_gate_type'],
            hsic_proj_dim=model_cfg['hsic_proj_dim'],
            hsic_alpha=model_cfg['hsic_alpha'],
            aiics_mode=model_cfg['aiics_mode'],
        )
    elif config.network.startswith('TopoMamba_2D_'):
        model = create_topomamba2d_model(
            model_cfg['model_name'],
            in_chans=model_cfg['input_channels'],
            num_classes=model_cfg['num_classes'],
            branch_mode=model_cfg.get('branch_mode', 'both'),
            enable_cache=model_cfg.get('enable_cache', True),
            enable_gating=model_cfg.get('enable_gating', True),
            fusion_method=model_cfg.get('fusion_method', 'hsic'),
            hsic_proj_dim=model_cfg.get('hsic_proj_dim', 32),
            hsic_alpha=model_cfg.get('hsic_alpha', 0.5),
            hsic_temperature=model_cfg.get('hsic_temperature', 1.5),
            hsic_residual=model_cfg.get('hsic_residual', 0.3),
            scan_cache_mode=model_cfg.get('scan_cache_mode', 'buffer'),
        )
    
    elif config.network == 'HBFormer':
        model = create_hbformer_model(model_cfg)
    elif config.network == 'SMAFormer':
        model = create_smaformer_lits_model(model_cfg)
    elif config.network == 'SMAFormer_V3':
        model = create_smaformer_v3_model(model_cfg)
    elif config.network == 'DWSegnet':
        model = create_dwsegnet_model(model_cfg)
    elif config.network == 'AFFSegnet':
        model = create_affsegnet_model(model_cfg)
        
    else:
        raise ValueError(f"Unsupported network: {config.network}")
    
    model = model.cuda()
    
    # 加载最佳权重
    best_weight_path = config.test_weights_path or f'results/{config.network}_synapse/checkpoints/best.pth'
    if os.path.exists(best_weight_path):
        print(f"🚀 加载测试权重: {best_weight_path}")
        best_weight = torch.load(best_weight_path, map_location=torch.device('cpu'))
        
        # 处理DataParallel模型的module.前缀问题
        if any(key.startswith('module.') for key in best_weight.keys()):
            print("🔧 检测到DataParallel权重，正在移除module.前缀...")
            new_state_dict = {}
            for key, value in best_weight.items():
                if key.startswith('module.'):
                    new_key = key[7:]  # 移除'module.'前缀
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            best_weight = new_state_dict
            print("✅ 权重键名修复完成")
        
        # 使用strict=False加载权重，允许部分不匹配
        missing_keys, unexpected_keys = model.load_state_dict(best_weight, strict=False)
        
        if missing_keys:
            print(f"⚠️  缺失的权重键: {len(missing_keys)} 个")
            # 只显示前几个关键的缺失键
            for key in missing_keys[:5]:
                print(f"   - {key}")
            if len(missing_keys) > 5:
                print(f"   ... 和其他 {len(missing_keys)-5} 个键")
                
        if unexpected_keys:
            print(f"⚠️  意外的权重键: {len(unexpected_keys)} 个")
            # 只显示前几个意外的键
            for key in unexpected_keys[:5]:
                print(f"   - {key}")
            if len(unexpected_keys) > 5:
                print(f"   ... 和其他 {len(unexpected_keys)-5} 个键")
        
        if not missing_keys and not unexpected_keys:
            print("✅ 权重加载完成，所有键完全匹配")
        else:
            print("⚠️  权重部分加载完成，存在键名不匹配")
    else:
        raise FileNotFoundError(f"❌ 找不到测试权重文件: {best_weight_path}")
    
    print('#----------开始测试----------#')
    
    # 使用带可视化功能的验证函数
    from engine_synapse import val_one_epoch_with_visualization
    
    mean_dice, mean_hd95, dice_per_organ, hd95_per_organ, std_dice, std_hd95 = val_one_epoch_with_visualization(
        val_dataset,
        val_loader,
        model,
        0,  # epoch=0 for testing
        logger,
        config,
        test_save_path=test_work_dir,
        prediction_vis_dir=comparison_dir,
        attention_vis_dir=attention_vis_dir,
        val_or_test=True,  # 测试阶段
        save_vis_every_n=1,  # 每个案例都保存可视化
        # 提供本地定义的可视化函数
        save_prediction_comparison_func=save_prediction_comparison,
        extract_attention_maps_func=extract_attention_maps,
        save_attention_heatmaps_func=save_attention_heatmaps,
        extract_activation_maps_func=extract_activation_maps,
        save_activation_heatmaps_func=save_activation_heatmaps,
        activation_vis_dir=activation_vis_dir  # 添加activation可视化目录
    )
    
    # 测试完成后清理内存
    torch.cuda.empty_cache()
    gc.collect()
    print("🧹 测试完成，已清理GPU缓存")
    
    # 确保获得了详细的器官评估指标
    organ_names = ['Aorta', 'Gallbladder', 'Left_Kidney', 'Right_Kidney', 'Liver', 'Pancreas', 'Spleen', 'Stomach']
    if len(dice_per_organ) == 0:
        print("⚠️ 警告：没有获得详细的器官Dice信息，使用默认值")
        dice_per_organ = [0.0] * len(organ_names)
        hd95_per_organ = [0.0] * len(organ_names)
    
    # 整理测试结果
    results = {
        'model_info': {
            'network': config.network,
            'model_config': dict(model_cfg),
            'test_weight_path': best_weight_path
        },
        'summary': {
            'avg_dice': float(mean_dice),
            'mean_hd95': float(mean_hd95),
            'std_dice': float(std_dice),
            'std_hd95': float(std_hd95),
            'dice_per_organ': dice_per_organ,
            'hd95_per_organ': hd95_per_organ
        },
        'organ_details': {
            organ_names[i]: {
                'dice': float(dice_per_organ[i]) if i < len(dice_per_organ) else 0.0,
                'hd95': float(hd95_per_organ[i]) if i < len(hd95_per_organ) else 0.0
            } for i in range(len(organ_names))
        },
        'test_info': {
            'test_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_test_cases': len(val_dataset),
            'output_dir': test_work_dir
        }
    }
    
    # 保存测试结果
    save_test_results(config, results, test_work_dir)
    
    print('\n' + '='*80)
    print('🎉 测试完成！')
    print('='*80)
    print(f"📊 平均Dice: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"📊 平均HD95: {mean_hd95:.4f} ± {std_hd95:.4f}")
    print(f"📁 结果保存路径: {test_work_dir}")
    print(f"🖼️  预测对比图: {comparison_dir}")
    print(f"🔥 注意力热图: {attention_vis_dir}")
    print(f"⚡ 激活热图: {activation_vis_dir}")
    print('='*80)


if __name__ == '__main__':
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Synapse测试程序 - 支持命令行参数覆盖配置')
    
    # 常用参数
    parser.add_argument('--network', type=str, default=None,
                       choices=['vmunet', 'cumamba_v3_t', 'cumamba_v3_s', 'cumamba_v3_b', 'TopoMamba_2D_t', 'TopoMamba_2D_s', 'TopoMamba_2D_b',
                               'TopoMamba_3D_t', 'topomamba_3d_t',
                               'dzzmamba_t', 'dzzmamba_s', 'dzzmamba_b', 'SwinUnet',
                               'HBFormer', 'SMAFormer', 'SMAFormer_V3', 'DWSegnet', 'AFFSegnet'],
                       help='模型类型')
    parser.add_argument('--weights', type=str, default=None,
                       help='权重文件路径')
    parser.add_argument('--num_workers', type=int, default=1,
                       help='数据加载工作进程数')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='批次大小')
    parser.add_argument('--mia_mode', type=str, default='hsic',
                       help='["cos", "hsic"]')
    add_shared_experiment_args(parser)
    
    args = parser.parse_args()
    
    # 加载默认配置
    config = setting_config
    
    # 使用命令行参数覆盖配置
    if args.network is not None:
        config.network = args.network
        # 根据网络类型更新模型配置
        network_config_map = {
            'vmunet': config.vmunet_config,
            'cumamba_v3_t': config.topomamba_2d_t_config,
            'cumamba_v3_s': config.topomamba_2d_s_config,
            'cumamba_v3_b': config.topomamba_2d_b_config,
            'TopoMamba_2D_t': config.topomamba_2d_t_config,
            'TopoMamba_2D_s': config.topomamba_2d_s_config,
            'TopoMamba_2D_b': config.topomamba_2d_b_config,
            'TopoMamba_3D_t': config.topomamba_3d_t_config,
            'topomamba_3d_t': config.topomamba_3d_t_config,
            'dzzmamba_t': config.topomamba_2d_t_config,
            'dzzmamba_s': config.topomamba_2d_s_config,
            'dzzmamba_b': config.topomamba_2d_b_config,
            'SwinUnet': config.vmunet_config,
            'HBFormer': config.vmunet_config,
            'SMAFormer': config.vmunet_config,
            'SMAFormer_V3': config.vmunet_config,
            'DWSegnet': config.vmunet_config,
            'AFFSegnet': config.vmunet_config,
        }
        config.model_config = network_config_map.get(args.network, config.model_config)
    
    if args.weights is not None:
        config.test_weights_path = args.weights
    
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    shared_changes = apply_shared_experiment_overrides(config, args)
    print_shared_override_report(shared_changes)
    
    # 打印测试配置信息
    print("=" * 80)
    print("🧪 测试配置:")
    print("=" * 80)
    print(f"  模型: {config.network}")
    print(f"  权重路径: {config.test_weights_path}")
    print(f"  数据工作进程: {config.num_workers}")
    print("=" * 80)

    if is_synapse3d_model(config.network):
        from tools.synapse3d_pipeline import run_synapse3d_test_from_config

        run_synapse3d_test_from_config(config)
    else:
        main() 
