import os
import sys
# 在导入任何模块之前设置环境变量来抑制torchvision警告
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1, 2, 3"

if __name__ == "__main__" and "--smoke" in sys.argv:
    from tools.smoke_entrypoints import run_smoke_from_config
    run_smoke_from_config("configs.config_setting_synapse", "train_synapse.py")


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
    from tools.synapse3d_pipeline import run_synapse3d_training_from_legacy_argv

    run_synapse3d_training_from_legacy_argv(sys.argv[1:])
    sys.exit(0)

import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from datasets.dataset import RandomGenerator, SwinUnet_RandomGenerator
from engine_synapse import *
from engine_synapse import val_one_epoch_with_visualization, train_one_epoch, val_one_epoch, val_one_epoch_fast_dice_only  # 明确导入所需函数
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
    from models.SwinUnet import SwinUnet, create_swinunet_model
except ImportError:
    SwinUnet = _missing_optional_model("SwinUnet", "models.SwinUnet")
    create_swinunet_model = _missing_optional_model("SwinUnet", "models.SwinUnet")
try:
    from models.HBFormer import HBFormer, create_hbformer_model
except ImportError:
    HBFormer = _missing_optional_model("HBFormer", "models.HBFormer")
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
try:
    from models.cumambaV3 import create_model as create_cumamba_model
except ImportError:
    create_cumamba_model = _missing_optional_model("cumambaV3", "models.cumambaV3")
from models.TopoMamba_2D import create_model as create_topomamba2d_model

import json
import csv
import logging
from datetime import datetime

from utils import *
from configs.config_setting_synapse import setting_config
from tools.experiment_overrides import (
    add_shared_experiment_args,
    apply_shared_experiment_overrides,
    print_shared_override_report,
)
from tools.synapse3d_registry import is_synapse3d_model

import warnings
warnings.filterwarnings("ignore")
# 专门忽略torchvision的CUDA扩展警告
warnings.filterwarnings("ignore", "Failed to load image Python extension")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
# 忽略所有UserWarning
warnings.filterwarnings("ignore", category=UserWarning)
# 忽略thop相关的INFO信息
logging.getLogger('thop').setLevel(logging.WARNING)

# 尝试导入thop，如果失败则设置标志
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("⚠️  警告: thop库未安装，将跳过FLOPs计算")

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
def _clear_thop_hooks(model: torch.nn.Module):
    """彻底清理 thop 遗留的 hooks 和统计属性，防止 CPU/CUDA 设备不一致。"""
    try:
        for m in model.modules():
            if hasattr(m, '_forward_hooks'):
                m._forward_hooks.clear()
            if hasattr(m, '_forward_pre_hooks'):
                m._forward_pre_hooks.clear()
            if hasattr(m, '_backward_hooks'):
                m._backward_hooks.clear()
            if hasattr(m, 'total_ops'):
                try:
                    delattr(m, 'total_ops')
                except Exception:
                    pass
            if hasattr(m, 'total_params'):
                try:
                    delattr(m, 'total_params')
                except Exception:
                    pass
    except Exception:
        pass

import numpy as np
import cv2
from scipy.ndimage import zoom


def save_train_record(config, epoch, lr, loss, avg_dice, dice_per_organ):
    """保存训练记录到CSV文件"""
    record_file = os.path.join(config.work_dir, 'train_record.csv')
    
    # 准备记录数据
    record = {
        'epoch': epoch,
        'lr': lr,
        'loss': loss,
        'avg_dice': avg_dice,
    }
    
    # 添加每个器官的dice
    organ_names = ['Aorta', 'Gallbladder', 'Left_Kidney', 'Right_Kidney', 'Liver', 'Pancreas', 'Spleen', 'Stomach']
    for i, organ in enumerate(organ_names):
        if i < len(dice_per_organ):
            record[f'dice_{organ}'] = dice_per_organ[i]
    
    # 检查文件是否存在，如果不存在则创建并写入头部
    file_exists = os.path.exists(record_file)
    
    with open(record_file, 'a', newline='') as f:
        fieldnames = ['epoch', 'lr', 'loss', 'avg_dice'] + [f'dice_{organ}' for organ in organ_names]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(record)
    
    print(f"训练记录已保存到: {record_file}")


def save_test_record(config, epoch, avg_dice, mean_hd95, dice_per_organ, hd95_per_organ):
    """保存测试记录到CSV文件"""
    record_file = os.path.join(config.work_dir, 'test_record.csv')
    
    # 准备记录数据
    record = {
        'epoch': epoch,
        'avg_dice': avg_dice,
        'mean_hd95': mean_hd95,
    }
    
    # 添加每个器官的dice和hd95
    organ_names = ['Aorta', 'Gallbladder', 'Left_Kidney', 'Right_Kidney', 'Liver', 'Pancreas', 'Spleen', 'Stomach']
    for i, organ in enumerate(organ_names):
        if i < len(dice_per_organ):
            record[f'dice_{organ}'] = dice_per_organ[i]
        if i < len(hd95_per_organ):
            record[f'hd95_{organ}'] = hd95_per_organ[i]
    
    # 检查文件是否存在，如果不存在则创建并写入头部
    file_exists = os.path.exists(record_file)
    
    with open(record_file, 'a', newline='') as f:
        fieldnames = ['epoch', 'avg_dice', 'mean_hd95'] + \
                    [f'dice_{organ}' for organ in organ_names] + \
                    [f'hd95_{organ}' for organ in organ_names]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(record)
    
    print(f"测试记录已保存到: {record_file}")


def check_pretrained_loading_vmunet(model, config):
    """检查VMUNet的预训练权重加载情况"""
    print("\n" + "="*80)
    print("🔍 检查 VMUNet 预训练权重加载情况")
    print("="*80)
    
    ckpt_path = config.model_config.get('load_ckpt_path')
    
    if not ckpt_path:
        print("❌ 未设置预训练权重路径 (load_ckpt_path)")
        return None
    
    if not os.path.exists(ckpt_path):
        print(f"❌ 预训练权重文件不存在: {ckpt_path}")
        return None
    
    print(f"📁 预训练权重路径: {ckpt_path}")
    
    try:
        # 加载预训练权重
        result = model.load_pretrained_weights(ckpt_path, verbose=True)
        
        # 检查返回值类型
        if isinstance(result, bool):
            # 如果是布尔值，表示简单的成功/失败
            if result:
                print(f"✅ 预训练权重加载成功")
                pretrained_info = {
                    'model_total_params': sum(p.numel() for p in model.parameters()),
                    'pretrained_path': ckpt_path,
                    'loading_method': 'load_pretrained_weights',
                    'status': 'success'
                }
            else:
                print(f"❌ 预训练权重加载失败")
                pretrained_info = {
                    'model_total_params': sum(p.numel() for p in model.parameters()),
                    'pretrained_path': ckpt_path,
                    'loading_method': 'load_pretrained_weights',
                    'status': 'failed'
                }
        elif isinstance(result, dict):
            # 如果是字典，包含详细信息
            print(f"✅ 预训练权重加载成功")
            encoder_info = result.get('encoder', {})
            decoder_info = result.get('decoder', {})
            
            pretrained_info = {
                'model_total_params': sum(p.numel() for p in model.parameters()),
                'pretrained_path': ckpt_path,
                'loading_method': 'load_pretrained_weights',
                'encoder_loaded': encoder_info.get('loaded_count', 0),
                'decoder_loaded': decoder_info.get('loaded_count', 0),
                'total_loaded': encoder_info.get('loaded_count', 0) + decoder_info.get('loaded_count', 0),
                'loading_ratio': result.get('loading_ratio', 0.0),
                'encoder_total_params': encoder_info.get('total_params', 0),
                'decoder_total_params': decoder_info.get('total_params', 0),
                'status': 'success'
            }
            
            print(f"🎯 总计加载: {pretrained_info['total_loaded']} 个权重键")
            print(f"📈 权重加载比例: {pretrained_info['loading_ratio']:.1f}%")
        else:
            # 未知返回类型
            print(f"⚠️  预训练权重加载返回了未知类型: {type(result)}")
            pretrained_info = {
                'model_total_params': sum(p.numel() for p in model.parameters()),
                'pretrained_path': ckpt_path,
                'loading_method': 'load_pretrained_weights',
                'status': 'unknown'
            }
        
        print("="*80)
        
        return pretrained_info
        
    except Exception as e:
        print(f"❌ 检查预训练权重失败: {e}")
        print("="*80)
        return None


def check_pretrained_loading_cumamba(model, config):
    """检查TopoMamba_2D系列模型的预训练权重加载情况"""
    print("\n" + "="*80)
    print(f"🔍 检查 {config.network.upper()} 预训练权重加载情况")
    print("="*80)
    
    load_pretrained = config.model_config.get('load_pretrained', False)
    pretrained_path = config.model_config.get('pretrained_path', '')
    
    if not load_pretrained:
        print("❌ 未启用预训练权重加载 (load_pretrained=False)")
        print("="*80)
        return None
    
    if not pretrained_path or not os.path.exists(pretrained_path):
        print(f"❌ 预训练权重文件不存在: {pretrained_path}")
        print("="*80)
        return None
    
    print(f"📁 预训练权重路径: {pretrained_path}")
    
    try:
        # 统计原始模型参数
        total_model_params = sum(p.numel() for p in model.parameters())
        encoder_total_params = sum(p.numel() for name, p in model.named_parameters() 
                                 if hasattr(model, '_is_encoder_param') and model._is_encoder_param(name))
        decoder_total_params = total_model_params - encoder_total_params
        
        # 加载预训练权重并收集详细信息
        if hasattr(model, 'load_pretrained_backbone'):
            print("🚀 正在加载预训练权重...")
            
            # 记录权重加载前的状态
            original_state = {}
            for name, param in model.named_parameters():
                original_state[name] = param.clone()
            
            # 执行权重加载
            result = model.load_pretrained_backbone(pretrained_path, verbose=False)
            
            # 分析实际加载情况
            loaded_params_detail = []
            not_loaded_params_detail = []
            encoder_loaded_count = 0
            decoder_loaded_count = 0
            
            for name, param in model.named_parameters():
                if not torch.equal(original_state[name], param):
                    # 参数被更新，说明成功加载
                    param_count = param.numel()
                    loaded_params_detail.append({
                        'name': name,
                        'shape': list(param.shape),
                        'param_count': param_count
                    })
                    
                    if hasattr(model, '_is_encoder_param') and model._is_encoder_param(name):
                        encoder_loaded_count += param_count
                    else:
                        decoder_loaded_count += param_count
                else:
                    # 参数未更新，说明未加载
                    param_count = param.numel()
                    not_loaded_params_detail.append({
                        'name': name,
                        'shape': list(param.shape),
                        'param_count': param_count
                    })
            
            total_loaded_params = encoder_loaded_count + decoder_loaded_count
            
            # 简化的终端显示
            print(f"📊 预训练权重加载统计:")
            print(f"   a. Total pretrained parameters: {total_loaded_params:,} ({total_loaded_params/1e6:.2f}M)")
            print(f"   b. Encoder loaded number: {encoder_loaded_count:,} ({encoder_loaded_count/1e6:.2f}M)")
            print(f"   c. Decoder loaded number: {decoder_loaded_count:,} ({decoder_loaded_count/1e6:.2f}M)")
            print("✅ 预训练权重加载完成！详细信息已保存到配置文件")
            
            return {
                'model_total_params': total_model_params,
                'encoder_total_params': encoder_total_params,
                'decoder_total_params': decoder_total_params,
                'encoder_loaded': encoder_loaded_count,
                'decoder_loaded': decoder_loaded_count,
                'total_loaded': total_loaded_params,
                'loading_ratio': (total_loaded_params / total_model_params) * 100,
                'pretrained_path': pretrained_path,
                'status': 'success',
                'loaded_params_detail': loaded_params_detail,
                'not_loaded_params_detail': not_loaded_params_detail
            }
        else:
            print("❌ 当前模型不支持预训练权重加载方法")
            return None
            
    except Exception as e:
        print(f"❌ 预训练权重加载出错: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        print("="*80)


def save_train_config(config, save_dir, model=None, pretrained_info=None):
    """保存训练配置信息到txt文件"""
    config_file = os.path.join(save_dir, 'train_config.txt')
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("🚀 TopoMamba_2D训练配置信息\n")
        f.write("=" * 80 + "\n")
        f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 模型配置
        f.write("📊 模型配置:\n")
        f.write(f"  网络类型: {config.network}\n")
        f.write(f"  模型名称: {config.model_config['model_name']}\n")
        f.write(f"  输入通道: {config.model_config['input_channels']}\n")
        f.write(f"  类别数量: {config.model_config['num_classes']}\n")
        f.write(f"  预训练权重: {config.model_config.get('pretrained_path', 'None')}\n")
        f.write(f"  加载预训练: {config.model_config.get('load_pretrained', False)}\n")
        
        # Mamba特有配置检测
        if model and ('mamba' in config.network.lower() or 'cumamba' in config.network.lower() or 'dzz' in config.network.lower()):
            f.write("\n🧬 Mamba模型特有配置:\n")
            try:
                # 尝试从模型的第一个SS2D_C块中获取配置
                mamba_config = {}
                for name, module in model.named_modules():
                    if 'self_attention' in name and hasattr(module, 'scan_cache_mode'):
                        mamba_config['scan_cache_mode'] = getattr(module, 'scan_cache_mode', 'unknown')
                        mamba_config['fusion_gate_type'] = getattr(module, 'fusion_gate_type', 'unknown')
                        mamba_config['branch_mode'] = getattr(module, 'branch_mode', 'unknown')
                        mamba_config['enable_cache'] = getattr(module, 'enable_cache', 'unknown')
                        mamba_config['enable_gating'] = getattr(module, 'enable_gating', 'unknown')
                        break
                
                if mamba_config:
                    f.write(f"  ScanCache模式: {mamba_config.get('scan_cache_mode', 'unknown')}\n")
                    f.write(f"  融合门控类型: {mamba_config.get('fusion_gate_type', 'unknown')}\n")
                    f.write(f"  分支模式: {mamba_config.get('branch_mode', 'unknown')}\n")
                    f.write(f"  启用缓存: {mamba_config.get('enable_cache', 'unknown')}\n")
                    f.write(f"  启用门控: {mamba_config.get('enable_gating', 'unknown')}\n")
                else:
                    f.write("  未找到SS2D_C配置信息\n")
            except Exception as e:
                f.write(f"  配置检测失败: {str(e)}\n")
        
        # 添加模型统计信息
        if model is not None:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_size_mb = total_params * 4 / (1024**2)
            
            f.write(f"  总参数数: {total_params:,} ({total_params/1e6:.2f}M)\n")
            f.write(f"  可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)\n")
            f.write(f"  模型大小: {model_size_mb:.2f} MB\n")
            
            # 计算FLOPs
            try:
                if THOP_AVAILABLE:
                    device = next(model.parameters()).device
                    input_tensor = torch.randn(1, config.model_config['input_channels'], 
                                             config.input_size_h, config.input_size_w).to(device)
                    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
                    f.write(f"  FLOPs: {flops:,} ({flops/1e9:.2f}G)\n")
                    print(f"📊 模型FLOPs: {flops/1e9:.2f}G")
                else:
                    f.write("  FLOPs: 跳过计算 (thop库未安装)\n")
                    print("⚠️  跳过FLOPs计算 (thop库未安装)")
                
            except Exception as e:
                print(f"⚠️  FLOPs计算失败: {e}")
                f.write(f"  FLOPs: 计算失败 ({str(e)[:100]})\n")
            finally:
                _clear_thop_hooks(model)
        
        # 添加预训练权重加载详情
        f.write("\n🏋️ 预训练权重加载详情:\n")
        if pretrained_info is None:
            f.write("  ❌ 未加载预训练权重或加载失败\n")
        else:
            f.write(f"  📁 预训练权重路径: {pretrained_info.get('pretrained_path', 'Unknown')}\n")
            
            if 'loaded_params_detail' in pretrained_info:
                # 新版详细统计信息
                f.write(f"  📊 模型总参数: {pretrained_info.get('model_total_params', 0):,} ({pretrained_info.get('model_total_params', 0)/1e6:.2f}M)\n")
                f.write(f"  💎 Encoder总参数: {pretrained_info.get('encoder_total_params', 0):,} ({pretrained_info.get('encoder_total_params', 0)/1e6:.2f}M)\n")
                f.write(f"  💎 Decoder总参数: {pretrained_info.get('decoder_total_params', 0):,} ({pretrained_info.get('decoder_total_params', 0)/1e6:.2f}M)\n")
                f.write(f"  🎯 总计加载参数: {pretrained_info.get('total_loaded', 0):,} ({pretrained_info.get('total_loaded', 0)/1e6:.2f}M)\n")
                f.write(f"  💎 Encoder加载参数: {pretrained_info.get('encoder_loaded', 0):,} ({pretrained_info.get('encoder_loaded', 0)/1e6:.2f}M)\n")
                f.write(f"  💎 Decoder加载参数: {pretrained_info.get('decoder_loaded', 0):,} ({pretrained_info.get('decoder_loaded', 0)/1e6:.2f}M)\n")
                f.write(f"  📈 权重加载比例: {pretrained_info.get('loading_ratio', 0):.1f}%\n")
                
                # 详细的权重加载报告
                f.write(f"\n{'='*60}\n")
                f.write("🔍 详细权重加载报告\n")
                f.write(f"{'='*60}\n")
                
                f.write("\n✅ 成功加载的权重层:\n")
                f.write("-" * 40 + "\n")
                for i, param_info in enumerate(pretrained_info['loaded_params_detail']):
                    f.write(f"{i+1:3d}. {param_info['name']}\n")
                    f.write(f"     形状: {param_info['shape']}\n")
                    f.write(f"     参数数: {param_info['param_count']:,}\n")
                f.write(f"\n总计成功加载: {len(pretrained_info['loaded_params_detail'])} 层\n")
                
                f.write("\n❌ 未加载的权重层:\n") 
                f.write("-" * 40 + "\n")
                for i, param_info in enumerate(pretrained_info['not_loaded_params_detail']):
                    f.write(f"{i+1:3d}. {param_info['name']}\n")
                    f.write(f"     形状: {param_info['shape']}\n")
                    f.write(f"     参数数: {param_info['param_count']:,}\n")
                f.write(f"\n总计未加载: {len(pretrained_info['not_loaded_params_detail'])} 层\n")
                
            elif 'pretrained_total_params' in pretrained_info:
                # VMUNet旧式风格的统计信息
                f.write(f"  📊 预训练权重总参数: {pretrained_info['pretrained_total_params']:,} ({pretrained_info['pretrained_total_params']/1e6:.2f}M)\n")
                f.write(f"  💎 Encoder加载参数: {pretrained_info['encoder_loaded_params']:,} ({pretrained_info['encoder_loaded_params']/1e6:.2f}M)\n")
                f.write(f"  💎 Decoder加载参数: {pretrained_info['decoder_loaded_params']:,} ({pretrained_info['decoder_loaded_params']/1e6:.2f}M)\n")
                f.write(f"  🎯 总计加载参数: {pretrained_info['total_loaded_params']:,} ({pretrained_info['total_loaded_params']/1e6:.2f}M)\n")
                f.write(f"  📈 参数加载比例: {pretrained_info['loading_ratio']:.1f}%\n")
                f.write(f"  🔑 Encoder加载键数: {pretrained_info['encoder_keys']}\n")
                f.write(f"  🔑 Decoder加载键数: {pretrained_info['decoder_keys']}\n")
            elif 'loading_method' in pretrained_info:
                # VMUNet简化风格的统计信息
                f.write(f"  📊 模型总参数: {pretrained_info.get('model_total_params', 0):,} ({pretrained_info.get('model_total_params', 0)/1e6:.2f}M)\n")
                f.write(f"  🔧 加载方法: {pretrained_info.get('loading_method', 'Unknown')}\n")
                f.write(f"  ✅ 加载状态: {pretrained_info.get('status', 'Unknown')}\n")
                f.write(f"  💡 说明: VMUNet使用内置的load_from()方法自动加载匹配的预训练权重\n")
            else:
                # 基于新的 _pretrained_load_report 格式
                if 'loaded_total_params' in pretrained_info:
                    f.write(f"  📊 模型总参数: {pretrained_info.get('model_total_params', 0):,} ({pretrained_info.get('model_total_params', 0)/1e6:.2f}M)\n")
                    f.write(f"  🎯 成功加载参数: {pretrained_info.get('loaded_total_params', 0):,} ({pretrained_info.get('loaded_total_params', 0)/1e6:.2f}M)\n")
                    f.write(f"  🔑 映射成功键数: {pretrained_info.get('num_mapped_keys', 0)}\n")
                    f.write(f"  🔑 总权重键数: {pretrained_info.get('total_keys', 0)}\n")
                    f.write(f"  📈 参数加载比例: {pretrained_info.get('loading_ratio_params', 0):.2f}%\n")
                else:
                    # 降级处理旧格式
                    f.write(f"  📊 模型总参数: {pretrained_info.get('model_total_params', 0):,} ({pretrained_info.get('model_total_params', 0)/1e6:.2f}M)\n")
                    f.write(f"  🎯 成功加载参数: {pretrained_info.get('total_loaded', 0)} 个权重键\n")
                    f.write(f"  📈 权重加载比例: {pretrained_info.get('loading_ratio', 0):.1f}%\n")
                
        f.write("\n")
        
        # 数据配置
        f.write("💾 数据配置:\n")
        f.write(f"  数据集: {config.datasets_name}\n")
        f.write(f"  输入尺寸: {config.input_size_h}x{config.input_size_w}\n")
        f.write(f"  Batch Size: {config.batch_size}\n")
        f.write(f"  Workers: {config.num_workers}\n")
        f.write(f"  数据路径: {config.data_path}\n")
        f.write(f"  验证路径: {config.volume_path}\n\n")
        
        # 训练配置
        f.write("🏋️ 训练配置:\n")
        f.write(f"  Epochs: {config.epochs}\n")
        f.write(f"  优化器: {config.opt}\n")
        f.write(f"  学习率: {config.lr}\n")
        f.write(f"  权重衰减: {config.weight_decay}\n")
        f.write(f"  调度器: {config.sch}\n")
        f.write(f"  混合精度: {config.amp}\n")
        f.write(f"  验证间隔: {config.val_interval}\n")
        f.write(f"  保存间隔: {getattr(config, 'save_interval', config.val_interval)}\n")
        f.write(f"  打印间隔: {config.print_interval}\n\n")
        
        # 损失函数配置
        f.write("📉 损失函数配置:\n")
        f.write(f"  损失函数: {type(config.criterion).__name__}\n")
        f.write(f"  损失权重: {config.loss_weight}\n\n")
        
        # 优化配置
        f.write("⚡ 性能优化配置:\n")
        f.write(f"  使用编译优化: {getattr(config, 'use_compile', False)}\n")
        f.write(f"  Pin Memory: {getattr(config, 'pin_memory', True)}\n")
        f.write(f"  分布式训练: {config.distributed}\n\n")
        
        # 调度器详细配置
        if config.sch == 'CosineAnnealingLR':
            f.write("📈 学习率调度器详细配置:\n")
            f.write(f"  类型: 余弦退火学习率\n")
            f.write(f"  T_max: {config.T_max}\n")
            f.write(f"  eta_min: {config.eta_min}\n")
            f.write(f"  last_epoch: {config.last_epoch}\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"✅ 训练配置已保存到: {config_file}")
    if model is not None:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 模型统计信息已记录: {total_params/1e6:.2f}M 参数")
    if pretrained_info is not None:
        if 'total_loaded_params' in pretrained_info:
            print(f"🏋️ 预训练权重信息已记录: {pretrained_info['total_loaded_params']/1e6:.2f}M 参数加载")
            if 'loading_ratio_params' in pretrained_info:
                print(f"🏋️  载入比例: {pretrained_info['loading_ratio_params']:.2f}%  (keys {pretrained_info.get('num_mapped_keys','?')}/{pretrained_info.get('total_keys','?')})")
        else:
            pass


def log_model_statistics(model, config, logger):
    """记录模型统计信息"""
    
    logger.info("=" * 60)
    logger.info("📊 模型统计信息")
    logger.info("=" * 60)
    
    # 计算模型参数和FLOP
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"🔢 模型参数统计:")
    logger.info(f"   总参数数: {total_params:,} ({total_params/1e6:.2f}M)")
    logger.info(f"   可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    logger.info(f"   不可训练参数: {total_params-trainable_params:,}")
    logger.info(f"   模型大小: {total_params * 4 / (1024**2):.2f} MB")
    
    # 详细统计各层参数
    logger.info(f"🏗️ 模型结构统计:")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        logger.info(f"   {name}: {module_params:,} 参数 ({module_params/1e6:.2f}M)")
    
    # Mamba特有配置记录到logger
    if 'mamba' in config.network.lower() or 'cumamba' in config.network.lower() or 'dzz' in config.network.lower():
        logger.info("🧬 Mamba模型特有配置:")
        try:
            mamba_config = {}
            for name, module in model.named_modules():
                if 'self_attention' in name and hasattr(module, 'scan_cache_mode'):
                    mamba_config['scan_cache_mode'] = getattr(module, 'scan_cache_mode', 'unknown')
                    mamba_config['fusion_gate_type'] = getattr(module, 'fusion_gate_type', 'unknown')
                    mamba_config['branch_mode'] = getattr(module, 'branch_mode', 'unknown')
                    mamba_config['enable_cache'] = getattr(module, 'enable_cache', 'unknown')
                    mamba_config['enable_gating'] = getattr(module, 'enable_gating', 'unknown')
                    break
            
            if mamba_config:
                logger.info(f"   ScanCache模式: {mamba_config.get('scan_cache_mode', 'unknown')}")
                logger.info(f"   融合门控类型: {mamba_config.get('fusion_gate_type', 'unknown')}")
                logger.info(f"   分支模式: {mamba_config.get('branch_mode', 'unknown')}")
                logger.info(f"   启用缓存: {mamba_config.get('enable_cache', 'unknown')}")
                logger.info(f"   启用门控: {mamba_config.get('enable_gating', 'unknown')}")
            else:
                logger.info("   未找到SS2D_C配置信息")
        except Exception as e:
            logger.info(f"   配置检测失败: {str(e)}")
    
    # 计算FLOPs
    try:
        if THOP_AVAILABLE:
            device = next(model.parameters()).device
            input_tensor = torch.randn(1, config.model_config['input_channels'], 
                                     config.input_size_h, config.input_size_w).to(device)
            flops, params = profile(model, inputs=(input_tensor,), verbose=False)
            logger.info(f"💫 FLOPs: {flops:,} ({flops/1e9:.2f}G)")
        else:
            logger.warning("⚠️  FLOPs计算失败 (thop库未安装)")
    except Exception as e:
        logger.warning(f"⚠️  FLOPs计算失败: {e}")
    finally:
        _clear_thop_hooks(model)
    
    logger.info("=" * 60)


def log_pretrained_loading_details(model, config, logger):
    """记录预训练权重加载详情"""
    if not config.model_config.get('load_pretrained', False):
        logger.info("⚠️  未启用预训练权重加载")
        return None
        
    pretrained_path = config.model_config.get('pretrained_path')
    if not pretrained_path or not os.path.exists(pretrained_path):
        logger.info(f"⚠️  预训练权重文件不存在: {pretrained_path}")
        return None
    
    logger.info("=" * 60)
    logger.info("🏋️ 预训练权重加载详情")
    logger.info("=" * 60)
    logger.info(f"📁 预训练权重路径: {pretrained_path}")
    
    try:
        if hasattr(model, 'load_pretrained_backbone'):
            # TopoMamba_2D系列模型有专门的加载函数
            result = model.load_pretrained_backbone(pretrained_path, verbose=False)
            
            logger.info(f"✅ 预训练权重加载结果:")
            logger.info(f"   成功加载参数: {result.get('loaded', 0)} 个")
            logger.info(f"   未加载参数: {result.get('not_loaded', 0)} 个")
            
            # 如果有更详细的统计信息
            if 'encoder' in result:
                logger.info(f"   Encoder加载: {result['encoder'].get('loaded', 0)} 个参数")
            if 'decoder' in result:
                logger.info(f"   Decoder加载: {result['decoder'].get('loaded', 0)} 个参数")
                
            return result
        else:
            logger.info("⚠️  当前模型不支持详细的预训练权重加载统计")
            return None
            
    except Exception as e:
        logger.error(f"❌ 预训练权重加载失败: {e}")
        return None
    finally:
        logger.info("=" * 60)


def create_visualization_dirs(outputs_dir):
    """创建可视化文件夹"""
    prediction_vis_dir = os.path.join(outputs_dir, 'prediction_visualization')
    attention_vis_dir = os.path.join(outputs_dir, 'attention_heatmaps')
    
    os.makedirs(prediction_vis_dir, exist_ok=True)
    os.makedirs(attention_vis_dir, exist_ok=True)
    
    return prediction_vis_dir, attention_vis_dir


def save_prediction_comparison(image, ground_truth, prediction, case_name, save_dir, slice_idx=None):
    """保存预测对比图：输入图像、ground truth、预测结果的三联图"""
    
    # 如果是3D数据，选择中间切片进行可视化
    if len(image.shape) == 3:
        slice_idx = slice_idx if slice_idx is not None else image.shape[0] // 2
        image_slice = image[slice_idx]
        gt_slice = ground_truth[slice_idx] 
        pred_slice = prediction[slice_idx]
    else:
        image_slice = image
        gt_slice = ground_truth
        pred_slice = prediction
    
    # 归一化图像到0-1范围
    if image_slice.max() > 1.1:
        image_slice = image_slice / 255.0
    
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 输入图像
    axes[0].imshow(image_slice, cmap='gray')
    axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Ground Truth
    axes[1].imshow(gt_slice, cmap='jet', alpha=0.7)
    axes[1].imshow(image_slice, cmap='gray', alpha=0.3)
    axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 预测结果
    axes[2].imshow(pred_slice, cmap='jet', alpha=0.7)
    axes[2].imshow(image_slice, cmap='gray', alpha=0.3)
    axes[2].set_title('Prediction', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    slice_suffix = f"_slice{slice_idx}" if slice_idx is not None else ""
    save_path = os.path.join(save_dir, f'{case_name}{slice_suffix}_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return save_path


def extract_attention_maps(model, input_tensor):
    """提取模型的注意力图"""
    attention_maps = []
    
    def hook_fn(module, input, output):
        # 对于Mamba模型，我们关注SS2D层的输出
        if hasattr(module, 'self_attention') or 'SS2D' in str(type(module)):
            # 获取注意力权重或特征图
            if isinstance(output, torch.Tensor):
                # 取平均注意力或特征响应
                attn = output.mean(dim=1, keepdim=True)  # 沿通道维度平均
                attention_maps.append(attn.detach().cpu())
    
    # 注册钩子
    hooks = []
    for name, module in model.named_modules():
        if 'VSSBlock' in str(type(module)) or 'SS2D' in str(type(module)):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
    # 前向传播
    with torch.no_grad():
        _ = model(input_tensor)
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    return attention_maps


def save_attention_heatmaps(image, attention_maps, case_name, save_dir, slice_idx=None):
    """保存注意力热图"""
    
    if len(image.shape) == 3:
        slice_idx = slice_idx if slice_idx is not None else image.shape[0] // 2
        image_slice = image[slice_idx]
    else:
        image_slice = image
    
    # 归一化图像
    if image_slice.max() > 1.1:
        image_slice = image_slice / 255.0
    
    # 创建多个注意力热图
    num_layers = min(len(attention_maps), 6)  # 最多显示6层
    if num_layers == 0:
        return None
        
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
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
        
        # 归一化注意力图
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        
        # 显示叠加的注意力热图
        axes[i].imshow(image_slice, cmap='gray')
        axes[i].imshow(attn_map, cmap='jet', alpha=0.6)
        axes[i].set_title(f'Layer {i+1} Attention', fontsize=12)
        axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(num_layers, 6):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    slice_suffix = f"_slice{slice_idx}" if slice_idx is not None else ""
    save_path = os.path.join(save_dir, f'{case_name}{slice_suffix}_attention.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return save_path


def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)
    
    # 创建可视化目录
    # prediction_vis_dir, attention_vis_dir = create_visualization_dirs(outputs)

    global logger
    logger = get_logger('train', log_dir)

    log_config_info(config, logger)



    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]# [0, 1, 2, 3]
    torch.cuda.empty_cache()
    
    # 检查CUDA可用性和GPU状态
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA不可用！请检查GPU环境")
    
    print(f"🔧 CUDA版本: {torch.version.cuda}")
    print(f"🔧 PyTorch版本: {torch.__version__}")
    
    gpus_type, gpus_num = torch.cuda.get_device_name(), torch.cuda.device_count()
    print(f"🚀 GPU设备: {gpus_type}")
    print(f"🚀 GPU数量: {gpus_num}")
    print(f"🚀 当前GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"🚀 设置GPU ID: {gpu_ids}")
    
    # 设置CUDA优化
    torch.backends.cudnn.benchmark = True  # 优化cudnn性能
    torch.backends.cudnn.deterministic = False  # 允许非确定性操作以获得更好性能
    
    if config.distributed:
        print('#----------Start DDP----------#')
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.manual_seed_all(config.seed)
        config.local_rank = torch.distributed.get_rank()



    print('#----------Preparing dataset----------#')
    
    # 根据网络类型选择相应的数据增强器
    if config.network == 'SwinUnet':
        # SwinUnet使用专门的数据增强器
        train_transform = transforms.Compose([
            SwinUnet_RandomGenerator(output_size=[config.input_size_h, config.input_size_w])
        ])
    else:
        # 其他模型使用通用数据增强器
        train_transform = transforms.Compose([
            RandomGenerator(output_size=[config.input_size_h, config.input_size_w])
        ])
    
    # 统一并加强数据增广：对所有非SwinUnet模型使用RandomGenerator（内部含rot/flip/rotate/resize），
    # 若后续需要更强增广，可在RandomGenerator中扩展颜色抖动等操作
    train_dataset = config.datasets(base_dir=config.data_path, list_dir=config.list_dir, split="train",
                            transform=train_transform)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if config.distributed else None
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size//gpus_num if config.distributed else config.batch_size, 
                                shuffle=(train_sampler is None),
                                pin_memory=True,
                                num_workers=config.num_workers,
                                sampler=train_sampler)

    val_dataset = config.datasets(base_dir=config.volume_path, split="test_vol", list_dir=config.list_dir)
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if config.distributed else None
    val_loader = DataLoader(val_dataset,
                                batch_size=1, # if config.distributed else config.batch_size,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers, 
                                sampler=val_sampler,
                                drop_last=True)

    


    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config
    pretrained_info = {}
    # 打印CONFIG中的预训练优先级设置
    _ppath = model_cfg.get('pretrained_path', None)
    _ppath_abs = os.path.abspath(_ppath) if _ppath else None
    print(f"[CONFIG] load_pretrained={model_cfg.get('load_pretrained', False)}  pretrained_path={_ppath_abs}")
    
    if config.network == 'vmunet':
        model = VMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'] if model_cfg['load_pretrained'] else None,
        )
        print("✅ VMUNet模型创建完成")
         
    elif config.network.startswith('dzzmamba'):
        # 规模自适应：为small/base注入SS2D_C dropout与HSIC投影维度
        attn_drop_rate = 0.1 if config.network in ['dzzmamba_s', 'dzzmamba_b'] else 0.0
        hsic_proj_dim = model_cfg.get('hsic_proj_dim', 32)
        model = create_dzz_model(
            model_cfg['model_name'], 
            in_chans=model_cfg['input_channels'], 
            num_classes=model_cfg['num_classes'],
            load_pretrained=model_cfg.get('load_pretrained', False),
            pretrained_path=model_cfg.get('pretrained_path', None),
            aiics_mode='buffer',
            mia_gate_type='hsic',
            enable_gating=True,
            enable_cache=True,
            hsic_proj_dim=hsic_proj_dim,
            attn_drop_rate=attn_drop_rate,
        )
        print(f"✅ DZZMamba模型创建完成: {model_cfg['model_name']}")
 
     
    elif config.network.startswith('cumamba_v3'):
        # 使用DZZMamba工厂（已为cumamba_v3_*注册别名），并在模型内部按配置自动加载预训练
        attn_drop_rate = 0.1 if config.network in ['cumamba_v3_s', 'cumamba_v3_b'] else 0.0
        model = create_dzz_model(
            model_cfg['model_name'], 
            in_chans=model_cfg['input_channels'], 
            num_classes=model_cfg['num_classes'],
            load_pretrained=model_cfg.get('load_pretrained', False),
            pretrained_path=model_cfg.get('pretrained_path', None),
            attn_drop_rate=attn_drop_rate,
        )
        print(f"✅ TopoMamba_2DV3模型创建完成(经DZZMamba别名): {model_cfg['model_name']}")
        
    elif config.network.startswith('TopoMamba_2D_'):
        # 原生TopoMamba_2D实现（支持ScanCache与多种融合对比方法）
        attn_drop_rate = 0.1 if config.network in ['TopoMamba_2D_s', 'TopoMamba_2D_b'] else 0.0
        fusion_method = model_cfg.get('fusion_method', 'hsic')
        model = create_topomamba2d_model(
            model_cfg['model_name'],
            in_chans=model_cfg['input_channels'],
            num_classes=model_cfg['num_classes'],
            load_pretrained=model_cfg.get('load_pretrained', False),
            pretrained_path=model_cfg.get('pretrained_path', None),
            # ScanCache/HSIC 参数
            scan_cache_mode=model_cfg.get('scan_cache_mode', 'buffer'),
            enable_gating=model_cfg.get('enable_gating', True),
            enable_cache=model_cfg.get('enable_cache', True),
            hsic_proj_dim=model_cfg.get('hsic_proj_dim', 32),
            hsic_alpha=model_cfg.get('hsic_alpha', 0.5),
            hsic_temperature=model_cfg.get('hsic_temperature', 1.5),
            hsic_residual=model_cfg.get('hsic_residual', 0.3),
            fusion_method=fusion_method,
            attn_drop_rate=attn_drop_rate,
        )
        print(f"✅ TopoMamba_2D模型创建完成: {model_cfg['model_name']} | fusion={fusion_method}")

    elif config.network == 'SwinUnet':
        # SwinUnet使用工厂函数创建
        model = create_swinunet_model(model_cfg)
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
        raise ValueError(f"❌ 不支持的网络类型: {config.network}")
    
    model = model.cuda()
    
    # VMUNet 仍保留其内部加载；DZZMamba/TopoMamba_2DV3 已在模型内部自动尝试加载，此处不再重复
    if config.network == 'vmunet' and model_cfg.get('load_pretrained', False) and model_cfg.get('pretrained_path'):
        pretrained_path = model_cfg['pretrained_path']
        if os.path.exists(pretrained_path):
            try:
                model.load_from_checkpoint(pretrained_path)
                print("✅ VMUNet预训练权重加载完成")
                pretrained_info['loaded_from'] = pretrained_path
                pretrained_info['method'] = 'VMUNet.load_from_checkpoint'
            except Exception as e:
                print(f"⚠️ VMUNet 预训练权重加载失败: {e}")
        else:
            print(f"⚠️ 预训练权重文件不存在: {pretrained_path}")
    
    # 保存训练配置到文件
    # 预训练加载报告
    pretrained_info = getattr(model, '_pretrained_load_report', None)
    save_train_config(config, config.work_dir, model, pretrained_info)
    
    # 记录详细的模型统计信息
    log_model_statistics(model, config, logger)
    
    # 记录预训练权重加载详情
    log_pretrained_loading_details(model, config, logger)

    if config.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = DDP(model, device_ids=[config.local_rank], output_device=config.local_rank)
    else:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])




    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()



    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    print(f"#----------调试信息----------#")
    print(f"✅ min_loss: {min_loss}")
    print(f"✅ start_epoch: {start_epoch}")
    print(f"✅ min_epoch: {min_epoch}")
    print(f"✅ resume_model 路径: {resume_model}")
    print(f"✅ 文件是否存在: {os.path.exists(resume_model)}")
    print(f"✅ config.resume_training: {config.resume_training}")



    if os.path.exists(resume_model) and config.resume_training:
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        
        # 过滤不匹配的SS2D-DZZ参数
        checkpoint_state_dict = checkpoint['model_state_dict']
        model_state_dict = model.module.state_dict()
        
        # SS2D-DZZ新增的参数关键词
        ss2d_dzz_params = [
            'A_log_dzz', 'D_dzz', 'x_proj_dzz', 'dt_proj_dzz', 
            'gate_linear', 'gate_norm'
        ]
        
        # 过滤掉不匹配的参数
        filtered_checkpoint = {}
        skipped_keys = []
        loaded_keys = []
        
        for key, value in checkpoint_state_dict.items():
            if key in model_state_dict:
                # 检查形状是否匹配
                if value.shape == model_state_dict[key].shape:
                    filtered_checkpoint[key] = value
                    loaded_keys.append(key)
                else:
                    print(f"⚠️  形状不匹配，跳过: {key} (checkpoint: {value.shape} vs model: {model_state_dict[key].shape})")
                    skipped_keys.append(f"{key} (形状不匹配)")
            else:
                # 检查是否是SS2D-DZZ新增的参数
                is_ss2d_dzz = any(dzz_param in key for dzz_param in ss2d_dzz_params)
                if is_ss2d_dzz:
                    print(f"🔥 跳过SS2D-DZZ新参数: {key}")
                    skipped_keys.append(f"{key} (SS2D-DZZ新参数)")
                else:
                    print(f"⚠️  键不存在，跳过: {key}")
                    skipped_keys.append(f"{key} (键不存在)")
        
        # 使用strict=False加载过滤后的权重
        missing_keys, unexpected_keys = model.module.load_state_dict(filtered_checkpoint, strict=False)
        
        print(f"✅ 成功加载 {len(loaded_keys)} 个参数")
        print(f"⚠️  跳过 {len(skipped_keys)} 个参数")
        
        # 安全加载优化器和调度器状态
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✅ 优化器状态加载成功")
        except Exception as e:
            print(f"⚠️  优化器状态加载失败 (模型参数变化导致): {e}")
            print("💡 将使用新的优化器状态重新开始")
        
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✅ 调度器状态加载成功")
        except Exception as e:
            print(f"⚠️  调度器状态加载失败: {e}")
            print("💡 将使用新的调度器状态重新开始")
        
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)
    elif os.path.exists(resume_model) and not config.resume_training:
        print('#----------跳过训练恢复----------#')
        print(f"🔍 检测到checkpoint文件: {resume_model}")
        print(f"⚠️  config.resume_training = False，跳过恢复，从epoch 1重新开始训练")
        print(f"💡 如需恢复训练，请在configs/config_setting_synapse.py中设置 resume_training = True")
    else:
        print('#----------从零开始训练----------#')
        print(f"🚀 没有检测到checkpoint文件，从epoch 1开始全新训练")

    print(f"✅ 执行到训练前检查点")
    print(f"✅ config.network: {config.network}")
    print(f"✅ 即将进入训练循环...")

    print('#----------Training----------#')
    
    # 根据网络类型选择不同的训练策略
    if config.network == 'vmunet':
        print("🔄 使用原版VMUNet训练逻辑")
        # VMUNet使用原版简化的训练逻辑，但添加指标记录
        for epoch in range(start_epoch, config.epochs + 1):
            torch.cuda.empty_cache()
            train_sampler.set_epoch(epoch) if config.distributed else None

            # 使用增强的train_one_epoch_with_metrics函数来记录指标
            loss, train_dice_per_organ, avg_train_dice = train_one_epoch_with_metrics(
                train_loader,
                model,
                criterion,
                optimizer,
                scheduler,
                epoch,
                logger,
                config,
                scaler=scaler
            )
            
            # 记录训练信息
            current_lr = optimizer.param_groups[0]['lr']
            save_train_record(config, epoch, current_lr, loss, avg_train_dice, train_dice_per_organ)

            if loss < min_loss:
                torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
                min_loss = loss
                min_epoch = epoch

            if epoch % config.val_interval == 0:
                # 判断是否为最后一次epoch，只有最后一次才计算hd95
                is_last_epoch = (epoch == config.epochs)
                
                if is_last_epoch:
                    print(f"🎯 最后一次验证 (epoch {epoch})，计算完整指标包括HD95...")
                    # 使用原版的val_one_epoch函数，包含HD95计算
                    mean_dice, mean_hd95 = val_one_epoch(
                            val_dataset,
                            val_loader,
                            model,
                            epoch,
                            logger,
                            config,
                            test_save_path=outputs,
                            val_or_test=False
                        )
                    print(f"✅ 最终验证结果 - Dice: {mean_dice:.4f}, HD95: {mean_hd95:.4f}")
                else:
                    print(f"⚡ 中间验证 (epoch {epoch})，仅计算Dice指标...")
                    # 使用快速验证，不计算HD95
                    mean_dice = val_one_epoch_fast_dice_only(
                            val_dataset,
                            val_loader,
                            model,
                            epoch,
                            logger,
                            config,
                            test_save_path=outputs,
                            val_or_test=False
                        )
                    mean_hd95 = 0.0  # 占位值
                    print(f"⚡ 快速验证结果 - Dice: {mean_dice:.4f} (HD95跳过)")

            # 独立的保存逻辑：按照 save_interval 保存权重与latest（与验证解耦）
            if epoch % getattr(config, 'save_interval', config.val_interval) == 0:
                state_to_save = model.module.state_dict()
                
                # 完整训练快照（用于resume）
                torch.save({
                    'epoch': epoch,
                    'min_loss': min_loss,
                    'min_epoch': min_epoch,
                    'loss': loss,
                    'model_state_dict': state_to_save,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(checkpoint_dir, f'epoch_{epoch}.pth'))
                
                torch.save({
                    'epoch': epoch,
                    'min_loss': min_loss,
                    'min_epoch': min_epoch,
                    'loss': loss,
                    'model_state_dict': state_to_save,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(checkpoint_dir, 'latest.pth'))
                
                print(f'💾 保存检查点: epoch_{epoch}.pth 和 latest.pth')
    elif config.network == 'SwinUnet':
        print("🔄 使用SwinUnet训练逻辑")
        # SwinUnet使用增强的训练逻辑
        for epoch in range(start_epoch, config.epochs + 1):
            print(f"开始第 {epoch} 轮训练...")

            torch.cuda.empty_cache()
            train_sampler.set_epoch(epoch) if config.distributed else None

            try:
                # 训练一个epoch，获取详细信息用于记录
                print(f"调用 train_one_epoch_with_metrics...")
                loss, train_dice_per_organ, avg_train_dice = train_one_epoch_with_metrics(
                    train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    logger,
                    config,
                    scaler=scaler
                )
                print(f"训练完成，损失: {loss:.4f}, 平均Dice: {avg_train_dice:.4f}")
                
                # 记录训练信息
                current_lr = optimizer.param_groups[0]['lr']
                save_train_record(config, epoch, current_lr, loss, avg_train_dice, train_dice_per_organ)

                # 以验证Dice为准保存best在验证环节；此处保留基于loss的兜底best
                if loss < min_loss:
                    torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best_by_loss.pth'))
                    min_loss = loss
                    min_epoch = epoch

                # 最后一轮必须执行val；其余按间隔
                if (epoch % config.val_interval == 0) or (epoch == config.epochs):
                    print(f"开始验证第 {epoch} 轮...")
                    try:
                        # 判断是否为最后一次epoch，只有最后一次才计算hd95
                        is_last_epoch = (epoch == config.epochs)
                        
                        if is_last_epoch:
                            print(f"🎯 最后一次验证 (epoch {epoch})，计算完整指标包括HD95...")
                            # 使用详细指标验证函数，但不生成可视化图像
                            mean_dice, mean_hd95, dice_per_organ, hd95_per_organ = val_one_epoch_with_detailed_metrics(
                                    val_dataset,
                                    val_loader,
                                    model,
                                    epoch,
                                    logger,
                                    config,
                                    test_save_path=outputs,
                                    val_or_test=False  # 验证阶段，不保存测试结果
                                )
                            # 基于Dice保存best
                            prev_best = getattr(config, 'best_val_dice', -1)
                            if mean_dice > prev_best:
                                torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
                                config.best_val_dice = mean_dice
                            # 保存测试记录
                            save_test_record(config, epoch, mean_dice, mean_hd95, dice_per_organ, hd95_per_organ)
                            print(f"✅ 最终验证结果 - Dice: {mean_dice:.4f}, HD95: {mean_hd95:.4f}")
                        else:
                            print(f"⚡ 中间验证 (epoch {epoch})，仅计算Dice指标...")
                            # 使用快速验证，不计算HD95和可视化
                            mean_dice = val_one_epoch_fast_dice_only(
                                    val_dataset,
                                    val_loader,
                                    model,
                                    epoch,
                                    logger,
                                    config,
                                    test_save_path=outputs,
                                    val_or_test=False
                                )
                            # 基于Dice保存best（快速评估）
                            prev_best = getattr(config, 'best_val_dice', -1)
                            if mean_dice > prev_best:
                                torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
                                config.best_val_dice = mean_dice
                            mean_hd95 = 0.0  # 占位值
                            dice_per_organ = [mean_dice] * 8  # 占位值
                            hd95_per_organ = [0.0] * 8  # 占位值
                            print(f"⚡ 快速验证结果 - Dice: {mean_dice:.4f} (HD95和可视化跳过)")
                        
                        print(f"验证完成，平均Dice: {mean_dice:.4f}")
                    except Exception as e:
                        print(f"验证过程出错: {e}")
                        import traceback
                        traceback.print_exc()
                
                # 独立的保存逻辑：按照 save_interval 保存权重与latest（与验证解耦）
                if epoch % getattr(config, 'save_interval', config.val_interval) == 0:
                    state_to_save = model.module.state_dict()
                    # 完整训练快照（用于resume）
                    torch.save({
                        'epoch': epoch,
                        'min_loss': min_loss,
                        'min_epoch': min_epoch,
                        'loss': loss,
                        'model_state_dict': state_to_save,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, os.path.join(checkpoint_dir, 'epoch_{epoch}.pth'))
                    
                    torch.save({
                        'epoch': epoch,
                        'min_loss': min_loss,
                        'min_epoch': min_epoch,
                        'loss': loss,
                        'model_state_dict': state_to_save,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, os.path.join(checkpoint_dir, 'latest.pth'))
                    
                    print(f'💾 保存检查点: epoch_{epoch}.pth 和 latest.pth')
                
                print(f"第 {epoch} 轮训练完成")
                    
            except Exception as e:
                print(f"第 {epoch} 轮训练出错: {e}")
                import traceback
                traceback.print_exc()
                break

        # 最终测试阶段
        if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
            print('#----------Training Completed----------#')
            print(f"✅ 训练完成！最佳模型已保存")
            print(f"📁 最佳权重: {os.path.join(checkpoint_dir, 'best.pth')}")
            print(f"📈 最佳损失: {min_loss:.4f} (epoch {min_epoch})")
            print(f"🔍 要进行测试，请运行: python test_synapse.py")
    else:
        print("🚗 使用TopoMamba_2D训练逻辑")
        print(f"✅ 即将开始TopoMamba_2D训练循环，epoch范围: {start_epoch} 到 {config.epochs}")
        # TopoMamba_2D等其他模型使用增强的训练逻辑
        for epoch in range(start_epoch, config.epochs + 1):
            print(f"开始第 {epoch} 轮训练...")

            torch.cuda.empty_cache()
            train_sampler.set_epoch(epoch) if config.distributed else None

            try:
                # 训练一个epoch，获取详细信息用于记录
                print(f"调用 train_one_epoch_with_metrics...")
                loss, train_dice_per_organ, avg_train_dice = train_one_epoch_with_metrics(
                    train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    logger,
                    config,
                    scaler=scaler
                )
                print(f"训练完成，损失: {loss:.4f}, 平均Dice: {avg_train_dice:.4f}")
                
                # 记录训练信息
                current_lr = optimizer.param_groups[0]['lr']
                save_train_record(config, epoch, current_lr, loss, avg_train_dice, train_dice_per_organ)

                if loss < min_loss:
                    torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best_by_loss.pth'))
                    min_loss = loss
                    min_epoch = epoch

                if (epoch % config.val_interval == 0) or (epoch == config.epochs):
                    print(f"开始验证第 {epoch} 轮...")
                    try:
                        # 判断是否为最后一次epoch，只有最后一次才计算hd95
                        is_last_epoch = (epoch == config.epochs)
                        
                        if is_last_epoch:
                            print(f"🎯 最后一次验证 (epoch {epoch})，计算完整指标包括HD95...")
                            # 使用详细指标验证函数，但不生成可视化图像
                            mean_dice, mean_hd95, dice_per_organ, hd95_per_organ = val_one_epoch_with_detailed_metrics(
                                    val_dataset,
                                    val_loader,
                                    model,
                                    epoch,
                                    logger,
                                    config,
                                    test_save_path=outputs,
                                    val_or_test=False  # 验证阶段，不保存测试结果
                                )
                            prev_best = getattr(config, 'best_val_dice', -1)
                            if mean_dice > prev_best:
                                torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
                                config.best_val_dice = mean_dice
                            # 保存测试记录
                            save_test_record(config, epoch, mean_dice, mean_hd95, dice_per_organ, hd95_per_organ)
                            print(f"✅ 最终验证结果 - Dice: {mean_dice:.4f}, HD95: {mean_hd95:.4f}")
                        else:
                            print(f"⚡ 中间验证 (epoch {epoch})，仅计算Dice指标...")
                            # 使用快速验证，不计算HD95和可视化
                            mean_dice = val_one_epoch_fast_dice_only(
                                    val_dataset,
                                    val_loader,
                                    model,
                                    epoch,
                                    logger,
                                    config,
                                    test_save_path=outputs,
                                    val_or_test=False
                                )
                            prev_best = getattr(config, 'best_val_dice', -1)
                            if mean_dice > prev_best:
                                torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
                                config.best_val_dice = mean_dice
                            mean_hd95 = 0.0  # 占位值
                            dice_per_organ = [mean_dice] * 8  # 占位值
                            hd95_per_organ = [0.0] * 8  # 占位值
                            print(f"⚡ 快速验证结果 - Dice: {mean_dice:.4f} (HD95和可视化跳过)")
                        
                        print(f"验证完成，平均Dice: {mean_dice:.4f}")
                    except Exception as e:
                        print(f"验证过程出错: {e}")
                        import traceback
                        traceback.print_exc()
                
                # 独立的保存逻辑：按照 save_interval 保存权重与latest（与验证解耦）
                if epoch % getattr(config, 'save_interval', config.val_interval) == 0:
                    state_to_save = model.module.state_dict()
                    # 完整训练快照（用于resume）
                    torch.save({
                        'epoch': epoch,
                        'min_loss': min_loss,
                        'min_epoch': min_epoch,
                        'loss': loss,
                        'model_state_dict': state_to_save,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, os.path.join(checkpoint_dir, 'epoch_{epoch}.pth'))
                    
                    torch.save({
                        'epoch': epoch,
                        'min_loss': min_loss,
                        'min_epoch': min_epoch,
                        'loss': loss,
                        'model_state_dict': state_to_save,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, os.path.join(checkpoint_dir, 'latest.pth'))
                    
                    print(f'💾 保存检查点: epoch_{epoch}.pth 和 latest.pth')
                
                print(f"第 {epoch} 轮训练完成")
                    
            except Exception as e:
                print(f"第 {epoch} 轮训练出错: {e}")
                import traceback
                traceback.print_exc()
                break

        # 最终测试阶段
        if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
            print('#----------Training Completed----------#')
            print(f"✅ 训练完成！最佳模型已保存")
            print(f"📁 最佳权重: {os.path.join(checkpoint_dir, 'best.pth')}")
            print(f"📈 最佳损失: {min_loss:.4f} (epoch {min_epoch})")
            print(f"🔍 要进行测试，请运行: python test_synapse.py")


if __name__ == '__main__':
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Synapse训练程序 - 支持命令行参数覆盖配置')
    
    # 常用参数
    parser.add_argument('--network', type=str, default=None,
                       choices=['vmunet', 'TopoMamba_2D_t', 'TopoMamba_2D_s', 'TopoMamba_2D_b',
                               'TopoMamba_3D_t', 'topomamba_3d_t',
                               'dzzmamba_t', 'dzzmamba_s', 'dzzmamba_b', 'SwinUnet',
                               'HBFormer', 'SMAFormer', 'SMAFormer_V3', 'DWSegnet', 'AFFSegnet'],
                       help='模型类型')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='if GPU==409024G: HBFormer(30M,256x256): 25, TopoMamba_2D_t(29M,512x512): 6, ' 
                           ' TopoMamba_2D_s(512x512): 2, DZZMamba_s(512x512): 2,'
                           ' TopoMamba_2D_b(512x512): 1, DZZMamba_b(512x512): 1,'
                       'elif GPU==A600048G: ...(TODO)'
                       )
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=None,
                       help='学习率')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='数据加载工作进程数')
    parser.add_argument('--val_interval', type=int, default=None,
                       help='验证间隔（epoch）')
    parser.add_argument('--save_interval', type=int, default=None,
                       help='保存间隔（epoch）')
    parser.add_argument('--resume', type=lambda x: x.lower() == 'true', default=None,
                       help='是否恢复训练 (true/false)')
    parser.add_argument('--amp', type=lambda x: x.lower() == 'true', default=None,
                       help='是否使用混合精度训练 (true/false)')
    parser.add_argument('--work_dir', type=str, default=None,
                       help='工作目录路径')
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
            'SMAFormer_V3': config.vmunet_config,  # 复用通用字段
            'DWSegnet': config.vmunet_config,
            'AFFSegnet': config.vmunet_config,
        }
        config.model_config = network_config_map.get(args.network, config.model_config)
        # 更新工作目录
        if args.work_dir is None:
            config.work_dir = f'results/{args.network}_synapse/'
    
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.lr is not None:
        config.lr = args.lr
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.val_interval is not None:
        config.val_interval = args.val_interval
    if args.save_interval is not None:
        config.save_interval = args.save_interval
    if args.resume is not None:
        config.resume_training = args.resume
    if args.amp is not None:
        config.amp = args.amp
    if args.work_dir is not None:
        config.work_dir = args.work_dir
    shared_changes = apply_shared_experiment_overrides(config, args)
    print_shared_override_report(shared_changes)
    
    # 打印配置信息
    print("=" * 80)
    print("🚀 训练配置:")
    print("=" * 80)
    print(f"  模型: {config.network}")
    print(f"  批次大小: {config.batch_size}")
    print(f"  训练轮数: {config.epochs}")
    print(f"  学习率: {config.lr}")
    print(f"  数据工作进程: {config.num_workers}")
    print(f"  验证间隔: {config.val_interval}")
    print(f"  保存间隔: {config.save_interval}")
    print(f"  恢复训练: {config.resume_training}")
    print(f"  混合精度: {config.amp}")
    print(f"  工作目录: {config.work_dir}")
    print("=" * 80)

    if is_synapse3d_model(config.network):
        from tools.synapse3d_pipeline import run_synapse3d_training_from_config

        run_synapse3d_training_from_config(config)
    else:
        main(config)       
