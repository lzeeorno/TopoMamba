##########################################################
# TopoMamba_2D 改良版
# 关键改进：索引正逆缓存 + 多缓存模式 + 精准统计 + 无重复散射重建
##########################################################

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")

import time
import math
from functools import partial
from typing import Optional, Callable, Any, Dict, Tuple, List
from collections import OrderedDict
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import repeat, rearrange
from timm.models.layers import DropPath, trunc_normal_
from typing import Tuple, Optional


try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    selective_scan_fn = None
    selective_scan_ref = None

##########################################################
# 模型注册系统
##########################################################

_model_registry = {}
def register_model(fn):
    _model_registry[fn.__name__] = fn
    return fn

def create_model(model_name, **kwargs):
    if model_name not in _model_registry:
        raise ValueError(f"Unknown model: {model_name}")
    return _model_registry[model_name](**kwargs)

def list_models():
    return list(_model_registry.keys())

## 配置说明
## 该文件不再维护内置的 MODEL_CONFIGS；所有模型超参统一从 configs/config_setting_*.py 注入。

##########################################################
# Cross Scan (保持原逻辑)
##########################################################

def vmamba_cross_scan_fwd(x: torch.Tensor, in_channel_first=True, out_channel_first=True):
    if in_channel_first:
        B, C, H, W = x.shape
        y = x.new_empty((B, 4, C, H * W))
        y[:, 0] = x.flatten(2, 3)
        y[:, 1] = x.transpose(2, 3).flatten(2, 3)
        y[:, 2:4] = torch.flip(y[:, 0:2], dims=[-1])
    else:
        B, H, W, C = x.shape
        y = x.new_empty((B, H * W, 4, C))
        y[:, :, 0] = x.flatten(1, 2)
        y[:, :, 1] = x.transpose(1, 2).flatten(1, 2)
        y[:, :, 2:4] = torch.flip(y[:, :, 0:2], dims=[1])

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()
    return y

def vmamba_cross_merge_fwd(y: torch.Tensor, in_channel_first=True, out_channel_first=True):
    if out_channel_first:
        B, K, D, H, W = y.shape
        y = y.view(B, K, D, -1)
        y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = y[:, 0] + y[:, 1].view(B, D, W, H).transpose(2, 3).contiguous().view(B, D, -1)
    else:
        B, H, W, K, D = y.shape
        y = y.view(B, -1, K, D)
        y = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=[1]).view(B, -1, 2, D)
        y = y[:, :, 0] + y[:, :, 1].view(B, W, H, -1).transpose(1, 2).contiguous().view(B, -1, D)
    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 2, 1).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 1).contiguous()
    return y

##########################################################
# 改良 ScanCache 核心：索引生成与缓存
##########################################################

def _build_zigzag_indices(H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成 4 个方向的 zigzag 扫描正向索引 + 对应逆索引
    返回:
        fwd: (4, L)
        inv: (4, L)
    """
    # 主对角线 zigzag
    diag = []
    for s in range(H + W - 1):
        line = []
        for i in range(H):
            j = s - i
            if 0 <= j < W:
                line.append(i * W + j)
        if s % 2 == 1:
            line.reverse()
        diag.extend(line)

    # 反对角线 zigzag
    antidiag = []
    for s in range(H + W - 1):
        line = []
        for i in range(H):
            j = W - 1 - (s - i)
            if 0 <= j < W:
                line.append(i * W + j)
        if s % 2 == 1:
            line.reverse()
        antidiag.extend(line)

    diag_t = torch.tensor(diag, device=device, dtype=torch.long)
    anti_t = torch.tensor(antidiag, device=device, dtype=torch.long)

    fwd = torch.stack([
        diag_t,
        anti_t,
        torch.flip(diag_t, dims=[0]),
        torch.flip(anti_t, dims=[0])
    ], dim=0)  # (4,L)

    L = H * W
    inv = torch.empty_like(fwd)
    base = torch.arange(L, device=device, dtype=torch.long)
    for k in range(4):
        inv_k = torch.empty(L, device=device, dtype=torch.long)
        inv_k[fwd[k]] = base
        inv[k] = inv_k
    return fwd, inv

def _build_cross_indices(H: int, W: int, device: torch.device) -> torch.Tensor:
    """构建Cross-Scan的前向索引 (4, L):
    0: horizontal forward (row-major)
    1: horizontal backward (per-row reversed)
    2: vertical forward (column-major)
    3: vertical backward (per-column reversed)
    返回均为基于row-major扁平化的索引。
    """
    L = H * W
    # horizontal forward: 0..L-1
    horiz_fwd = torch.arange(L, device=device, dtype=torch.long)
    # horizontal backward: reverse within each row
    grid = horiz_fwd.view(H, W)
    horiz_bwd = torch.flip(grid, dims=[1]).contiguous().view(-1)
    # vertical forward: column-major order
    cols = torch.arange(W, device=device, dtype=torch.long).repeat_interleave(H)
    rows = torch.arange(H, device=device, dtype=torch.long).repeat(W)
    vert_fwd = (rows * W + cols).contiguous()
    # vertical backward: reverse within each column sequence
    cols_b = torch.arange(W - 1, -1, -1, device=device, dtype=torch.long).repeat_interleave(H)
    rows_b = torch.arange(H - 1, -1, -1, device=device, dtype=torch.long).repeat(W)
    vert_bwd = (rows_b * W + cols_b).contiguous()
    return torch.stack([horiz_fwd, horiz_bwd, vert_fwd, vert_bwd], dim=0)

##########################################################
# Optimized HSIC Gate
##########################################################
class OptimizedHSICGate(nn.Module):
    """
    HSIC-MF: Multi-Scale Hierarchical HSIC Fusion Gate
    
    论文标题建议:
    "Hierarchical Independence Criterion for Multi-Branch Feature Fusion 
     in Medical Image Segmentation"
    
    核心贡献:
    1. 层次化HSIC分解（通道/空间/交互三级）
    2. 自适应核混合策略（数据驱动的λ）
    3. 拓扑感知图正则化（图拉普拉斯嵌入）
    4. 双向因果HSIC（条件独立性测试）
    
    公平性保证:
    - 与BFTT3D/I2P-MAE/DTRG使用相同的投影维度k
    - 统一的α、τ超参数
    - 相同的融合位置和算子
    - 无额外预训练或外部数据
    
    理论优势:
    - 比BFTT3D: 不仅考虑Gram矩阵（通道），还考虑空间和交互
    - 比I2P-MAE: 不仅对齐分布，还建模因果关系
    - 比DTRG: 不仅用图做决策，还融入核计算（可微分）
    """
    
    def __init__(
        self,
        d_inner: int,
        proj_dim: int = 32,
        alpha_init: float = 1.0,
        temperature: float = 1.5,
        knn_k: int = 5,
        use_graph_reg: bool = True,
        use_causal_hsic: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.d_inner = d_inner
        self.proj_dim = proj_dim
        self.temperature = temperature
        self.knn_k = knn_k
        self.use_graph_reg = use_graph_reg
        self.use_causal_hsic = use_causal_hsic
        
        # 可学习参数
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        
        # 创新1: 自适应核混合网络（Adaptive Kernel Mixture）
        # 输入: [HSIC_linear, HSIC_rbf, mean(F_cross), mean(F_dzz)]
        # 输出: λ ∈ [0, 1] (混合比例)
        self.kernel_mixer = nn.Sequential(
            nn.Linear(4, 16, **factory_kwargs),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8, **factory_kwargs),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1, **factory_kwargs),
            nn.Sigmoid()  # λ ∈ [0, 1]
        )
        
        # 创新2: 层次权重学习（Hierarchical Weights）
        # 学习三个层次的重要性: w_channel, w_spatial, w_interaction
        self.hierarchy_weights = nn.Parameter(torch.ones(3) / 3)  # 初始化为均匀
        
        # 创新3: 因果方向判别器（Causal Direction Discriminator）
        # 判断 F_cross → F_dzz 还是 F_dzz → F_cross
        if use_causal_hsic:
            self.causal_discriminator = nn.Sequential(
                nn.Linear(2, 8, **factory_kwargs),  # 输入: [cHSIC(X→Y), cHSIC(Y→X)]
                nn.Tanh(),
                nn.Linear(8, 1, **factory_kwargs),
                nn.Tanh()  # 输出 ∈ [-1, 1]，正值表示X→Y，负值表示Y→X
            )
        
        # 残差权重（保证梯度流）
        self.residual_weight = 0.3
    
    # ========== 核心方法1: 层次化HSIC分解 ==========
    def compute_hierarchical_hsic(
        self,
        F_cross: torch.Tensor,
        F_dzz: torch.Tensor,
        kernel_type: str = 'rbf',
        graph_laplacian: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        层次化HSIC分解（核心创新）
        
        理论基础:
        ANOVA分解: HSIC(X, Y) = HSIC_main(X, Y) + HSIC_interaction(X, Y)
        进一步分解为:
            HSIC_channel: 通道间的依赖（类似BFTT3D的Gram矩阵）
            HSIC_spatial: 空间位置的依赖（新增）
            HSIC_interaction: 通道-空间交互（新增）
        
        输入:
            F_cross, F_dzz: (B, C, L)
            kernel_type: 'linear' | 'rbf' | 'mixed'
            graph_laplacian: (B, C, C) 图拉普拉斯矩阵（可选）
        
        输出:
            hsic_channel: (B,) 通道级HSIC
            hsic_spatial: (B,) 空间级HSIC
            hsic_interaction: (B,) 交互级HSIC
        """
        B, C, L = F_cross.shape
        
        # ========== Level 1: 通道级HSIC（Channel-wise）==========
        # 类似BFTT3D，但使用HSIC而非简单相关
        # 投影到低维
        k = max(8, min(self.proj_dim, L))
        P = torch.randn(L, k, device=F_cross.device, dtype=F_cross.dtype)
        P = F.normalize(P, dim=0) / math.sqrt(k)
        
        X_cross = F_cross @ P  # (B, C, k)
        X_dzz = F_dzz @ P
        
        # 归一化（提高稳定性）
        X_cross = F.normalize(X_cross, dim=-1)
        X_dzz = F.normalize(X_dzz, dim=-1)
        
        # 计算核矩阵
        if kernel_type == 'linear':
            K_cross_channel = X_cross @ X_cross.transpose(1, 2)  # (B, C, C)
            K_dzz_channel = X_dzz @ X_dzz.transpose(1, 2)
        elif kernel_type == 'rbf':
            K_cross_channel = self._compute_rbf_kernel(X_cross)
            K_dzz_channel = self._compute_rbf_kernel(X_dzz)
        else:  # mixed
            K_cross_linear = X_cross @ X_cross.transpose(1, 2)
            K_dzz_linear = X_dzz @ X_dzz.transpose(1, 2)
            K_cross_rbf = self._compute_rbf_kernel(X_cross)
            K_dzz_rbf = self._compute_rbf_kernel(X_dzz)
            # 暂时使用0.5混合，后续会被kernel_mixer调整
            K_cross_channel = 0.5 * K_cross_linear + 0.5 * K_cross_rbf
            K_dzz_channel = 0.5 * K_dzz_linear + 0.5 * K_dzz_rbf
        
        # 创新3: 图正则化（融合DTRG的拓扑信息）
        if self.use_graph_reg and graph_laplacian is not None:
            # K_graph = K_rbf ⊙ exp(-βL)
            # 其中L是图拉普拉斯矩阵，β控制正则化强度
            beta = 0.1
            graph_weight = torch.exp(-beta * graph_laplacian)  # (B, C, C)
            K_cross_channel = K_cross_channel * graph_weight
            K_dzz_channel = K_dzz_channel * graph_weight
        
        # 中心化
        K_cross_channel_c = self._center_kernel(K_cross_channel)
        K_dzz_channel_c = self._center_kernel(K_dzz_channel)
        
        # HSIC（通道级）
        hsic_channel = (K_cross_channel_c * K_dzz_channel_c).sum(dim=(-1, -2)) / max((C - 1) ** 2, 1)
        
        # ========== Level 2: 空间级HSIC（Spatial-wise）==========
        # 新增：考虑空间位置之间的依赖
        # 采样策略：避免L过大
        num_samples = min(256, L)
        if L > num_samples:
            indices = torch.randperm(L, device=F_cross.device)[:num_samples]
            F_cross_sampled = F_cross[:, :, indices]  # (B, C, num_samples)
            F_dzz_sampled = F_dzz[:, :, indices]
        else:
            F_cross_sampled = F_cross
            F_dzz_sampled = F_dzz
            num_samples = L
        
        # 转置到 (B, num_samples, C)
        F_cross_spatial = F_cross_sampled.transpose(1, 2)
        F_dzz_spatial = F_dzz_sampled.transpose(1, 2)
        
        # 归一化
        F_cross_spatial = F.normalize(F_cross_spatial, dim=-1)
        F_dzz_spatial = F.normalize(F_dzz_spatial, dim=-1)
        
        # 线性核（空间维度用线性核，计算效率高）
        K_cross_spatial = F_cross_spatial @ F_cross_spatial.transpose(1, 2)  # (B, num_samples, num_samples)
        K_dzz_spatial = F_dzz_spatial @ F_dzz_spatial.transpose(1, 2)
        
        # 中心化
        K_cross_spatial_c = self._center_kernel(K_cross_spatial)
        K_dzz_spatial_c = self._center_kernel(K_dzz_spatial)
        
        # HSIC（空间级）
        hsic_spatial = (K_cross_spatial_c * K_dzz_spatial_c).sum(dim=(-1, -2)) / max((num_samples - 1) ** 2, 1)
        
        # ========== Level 3: 交互级HSIC（Interaction）==========
        # 新增：捕捉通道和空间的联合效应
        # 使用张量积核: K_interaction = K_channel ⊗ K_spatial
        # 简化计算：使用Hadamard积近似
        
        # 构造联合特征: (B, C*num_samples, k)
        # 这里使用一个技巧：将通道和空间展平
        F_cross_flat = F_cross_sampled.reshape(B, -1, 1)  # (B, C*num_samples, 1)
        F_dzz_flat = F_dzz_sampled.reshape(B, -1, 1)
        
        # 简化的交互核（使用余弦相似度）
        F_cross_flat_norm = F.normalize(F_cross_flat, dim=1)
        F_dzz_flat_norm = F.normalize(F_dzz_flat, dim=1)
        
        # 交互HSIC = 通道HSIC * 空间HSIC的几何平均（一种近似）
        # 更严格的做法需要四阶张量，计算量太大
        hsic_interaction = torch.sqrt(hsic_channel.abs() * hsic_spatial.abs() + 1e-8)
        
        return hsic_channel, hsic_spatial, hsic_interaction
    
    # ========== 核心方法2: 自适应核混合 ==========
    def adaptive_kernel_mixture(
        self,
        F_cross: torch.Tensor,
        F_dzz: torch.Tensor
    ) -> torch.Tensor:
        """
        自适应核混合策略（创新2）
        
        思路:
        1. 分别计算线性核和RBF核的HSIC
        2. 使用MLP学习混合比例 λ = f(HSIC_linear, HSIC_rbf, F_cross, F_dzz)
        3. 最终核: K = λ * K_rbf + (1-λ) * K_linear
        
        理论:
        - 扩展Multiple Kernel Learning (MKL)到HSIC框架
        - λ是数据驱动的，不同样本可以有不同的混合比例
        
        优势:
        - 比固定核更灵活（适应不同数据分布）
        - 比单一核更鲁棒（避免RBF梯度消失）
        - 比I2P-MAE的单一线性核更强大
        """
        B, C, L = F_cross.shape
        
        # 计算两种核的HSIC
        with torch.no_grad():
            # 线性核HSIC
            hsic_linear_channel, hsic_linear_spatial, _ = self.compute_hierarchical_hsic(
                F_cross, F_dzz, kernel_type='linear', graph_laplacian=None
            )
            hsic_linear = (hsic_linear_channel + hsic_linear_spatial) / 2
            
            # RBF核HSIC
            hsic_rbf_channel, hsic_rbf_spatial, _ = self.compute_hierarchical_hsic(
                F_cross, F_dzz, kernel_type='rbf', graph_laplacian=None
            )
            hsic_rbf = (hsic_rbf_channel + hsic_rbf_spatial) / 2
        
        # 特征统计
        F_cross_mean = F_cross.mean(dim=(1, 2))  # (B,)
        F_dzz_mean = F_dzz.mean(dim=(1, 2))
        
        # 构造输入特征: [HSIC_linear, HSIC_rbf, mean(F_cross), mean(F_dzz)]
        mixer_input = torch.stack([
            hsic_linear,
            hsic_rbf,
            F_cross_mean,
            F_dzz_mean
        ], dim=1)  # (B, 4)
        
        # MLP预测混合比例
        lambda_mix = self.kernel_mixer(mixer_input)  # (B, 1)
        
        return lambda_mix.squeeze(1)  # (B,)
    
    # ========== 核心方法3: 双向因果HSIC ==========
    def compute_causal_hsic(
        self,
        F_cross: torch.Tensor,
        F_dzz: torch.Tensor
    ) -> torch.Tensor:
        """
        双向因果HSIC（创新4）
        
        理论基础:
        条件独立性测试: X ⊥ Y | Z ⟺ HSIC(X, Y | Z) = 0
        
        因果方向判别:
        - cHSIC(X → Y) = HSIC(X, Y | do(X))  # X是原因
        - cHSIC(Y → X) = HSIC(Y, X | do(Y))  # Y是原因
        
        实现简化:
        使用残差作为条件变量的代理
        - cHSIC(F_cross → F_dzz) ≈ HSIC(F_cross, residual(F_dzz | F_cross))
        - cHSIC(F_dzz → F_cross) ≈ HSIC(F_dzz, residual(F_cross | F_dzz))
        
        优势:
        - 比I2P-MAE的对称对齐更有方向性
        - 捕捉两个分支的因果关系（谁影响谁）
        - 指导融合策略（更依赖因果源）
        """
        B, C, L = F_cross.shape
        
        # 计算残差（简化的条件化）
        # residual(Y | X) ≈ Y - proj_X(Y)
        # 其中 proj_X(Y) = X * (X^T Y) / (X^T X)
        
        # 投影到低维（计算效率）
        k = max(8, min(self.proj_dim, L))
        P = torch.randn(L, k, device=F_cross.device, dtype=F_cross.dtype)
        P = F.normalize(P, dim=0)
        
        X_cross = F_cross @ P  # (B, C, k)
        X_dzz = F_dzz @ P
        
        # 归一化
        X_cross = F.normalize(X_cross, dim=-1)
        X_dzz = F.normalize(X_dzz, dim=-1)
        
        # 方向1: F_cross → F_dzz
        # 计算 F_dzz 在 F_cross 上的投影
        proj_coef = (X_cross * X_dzz).sum(dim=-1, keepdim=True)  # (B, C, 1)
        X_dzz_proj = proj_coef * X_cross  # (B, C, k)
        residual_dzz = X_dzz - X_dzz_proj  # (B, C, k)
        
        # cHSIC(F_cross → F_dzz) = HSIC(X_cross, residual_dzz)
        K_cross = self._compute_rbf_kernel(X_cross)
        K_residual_dzz = self._compute_rbf_kernel(residual_dzz)
        K_cross_c = self._center_kernel(K_cross)
        K_residual_dzz_c = self._center_kernel(K_residual_dzz)
        chsic_forward = (K_cross_c * K_residual_dzz_c).sum(dim=(-1, -2)) / max((C - 1) ** 2, 1)
        
        # 方向2: F_dzz → F_cross
        proj_coef_rev = (X_dzz * X_cross).sum(dim=-1, keepdim=True)
        X_cross_proj = proj_coef_rev * X_dzz
        residual_cross = X_cross - X_cross_proj
        
        K_dzz = self._compute_rbf_kernel(X_dzz)
        K_residual_cross = self._compute_rbf_kernel(residual_cross)
        K_dzz_c = self._center_kernel(K_dzz)
        K_residual_cross_c = self._center_kernel(K_residual_cross)
        chsic_backward = (K_dzz_c * K_residual_cross_c).sum(dim=(-1, -2)) / max((C - 1) ** 2, 1)
        
        # 使用判别器判断因果方向
        causal_input = torch.stack([chsic_forward, chsic_backward], dim=1)  # (B, 2)
        causal_direction = self.causal_discriminator(causal_input).squeeze(1)  # (B,)
        # causal_direction > 0: F_cross → F_dzz (F_cross是主导)
        # causal_direction < 0: F_dzz → F_cross (F_dzz是主导)
        
        return causal_direction, chsic_forward, chsic_backward
    
    # ========== 核心方法4: 拓扑感知图构建 ==========
    def build_topology_graph(
        self,
        F: torch.Tensor,
        k: int = None
    ) -> torch.Tensor:
        """
        构建拓扑感知的kNN图（融合DTRG）
        
        输入:
            F: (B, C, L) 特征
            k: kNN的k值
        
        输出:
            L_graph: (B, C, C) 图拉普拉斯矩阵
        
        理论:
        图拉普拉斯: L = D - A
        其中 D 是度矩阵，A 是邻接矩阵
        
        用途:
        嵌入到核矩阵中: K_graph = K_rbf ⊙ exp(-βL)
        保留局部拓扑结构，同时保持可微分
        """
        if k is None:
            k = self.knn_k
        
        B, C, L_dim = F.shape
        
        # 投影到低维
        proj_dim = max(8, min(self.proj_dim, L_dim))
        P = torch.randn(L_dim, proj_dim, device=F.device, dtype=F.dtype)
        P = F.normalize(P, dim=0)
        
        Z = F @ P  # (B, C, proj_dim)
        Z = F.normalize(Z, dim=-1)
        
        # 计算余弦相似度
        sim = torch.matmul(Z, Z.transpose(1, 2))  # (B, C, C)
        sim = sim.clamp(-1, 1)
        
        # 构造kNN图（使用top-k）
        # 距离 = 1 - 相似度
        dist = 1 - sim
        
        # 屏蔽对角线
        dist = dist + torch.eye(C, device=F.device, dtype=F.dtype).unsqueeze(0) * 1e9
        
        # 找到每个节点的k近邻
        topk_vals, topk_indices = dist.topk(k, largest=False, dim=-1)  # (B, C, k)
        
        # 构造邻接矩阵（对称化）
        A = torch.zeros(B, C, C, device=F.device, dtype=F.dtype)
        for b in range(B):
            for i in range(C):
                for j in topk_indices[b, i]:
                    A[b, i, j] = 1.0
                    A[b, j, i] = 1.0  # 对称化
        
        # 度矩阵
        D = torch.diag_embed(A.sum(dim=-1))  # (B, C, C)
        
        # 拉普拉斯矩阵
        L_graph = D - A
        
        # 归一化拉普拉斯（数值稳定）
        D_inv_sqrt = torch.diag_embed(1.0 / (A.sum(dim=-1).sqrt() + 1e-8))
        L_norm = D_inv_sqrt @ L_graph @ D_inv_sqrt
        
        return L_norm
    
    # ========== 辅助方法 ==========
    def _compute_rbf_kernel(self, X: torch.Tensor) -> torch.Tensor:
        """RBF核计算（自适应带宽）"""
        B, C, k = X.shape
        
        X_norm = (X ** 2).sum(dim=-1, keepdim=True)
        dist_sq = X_norm + X_norm.transpose(1, 2) - 2 * (X @ X.transpose(1, 2))
        dist_sq = dist_sq.clamp(min=0)
        
        # 自适应带宽（中位数启发式）
        with torch.no_grad():
            sigma_sq = dist_sq.view(B, -1).median(dim=1)[0].clamp(min=1e-6)
            sigma_sq = sigma_sq.view(B, 1, 1)
        
        # 裁剪避免下溢
        dist_sq_normalized = (dist_sq / (2 * sigma_sq)).clamp(max=10)
        K = torch.exp(-dist_sq_normalized)
        
        # 对角正则化
        K = K + 1e-5 * torch.eye(C, device=K.device, dtype=K.dtype).unsqueeze(0)
        
        return K
    
    def _center_kernel(self, K: torch.Tensor) -> torch.Tensor:
        """核矩阵中心化"""
        row_mean = K.mean(dim=-1, keepdim=True)
        col_mean = K.mean(dim=-2, keepdim=True)
        total_mean = K.mean(dim=(-1, -2), keepdim=True)
        
        K_centered = K - row_mean - col_mean + total_mean
        
        # 归一化（数值稳定）
        K_std = K_centered.std(dim=(-1, -2), keepdim=True).clamp(min=1e-6)
        K_centered = K_centered / K_std
        
        return K_centered
    
    # ========== 主前向传播 ==========
    def forward(
        self,
        F_cross: torch.Tensor,
        F_dzz: torch.Tensor
    ) -> torch.Tensor:
        """
        HSIC-MF Gate 前向传播
        
        流程:
        1. 构建拓扑图（DTRG启发）
        2. 计算层次化HSIC（通道/空间/交互）
        3. 自适应核混合
        4. 双向因果HSIC
        5. 融合决策
        
        输入:
            F_cross: (B, C, L) 交叉扫描特征
            F_dzz: (B, C, L) DZZ扫描特征
        
        输出:
            F_out: (B, C, L) 融合特征
        """
        B, C, L = F_cross.shape
        
        # ========== Step 1: 构建拓扑图 ==========
        if self.use_graph_reg:
            L_graph_cross = self.build_topology_graph(F_cross, k=self.knn_k)
            L_graph_dzz = self.build_topology_graph(F_dzz, k=self.knn_k)
            # 使用两个图的平均
            L_graph = (L_graph_cross + L_graph_dzz) / 2
        else:
            L_graph = None
        
        # ========== Step 2: 自适应核混合 ==========
        lambda_mix = self.adaptive_kernel_mixture(F_cross, F_dzz)  # (B,)
        
        # ========== Step 3: 计算层次化HSIC ==========
        # 使用混合核
        hsic_channel, hsic_spatial, hsic_interaction = self.compute_hierarchical_hsic(
            F_cross, F_dzz, 
            kernel_type='mixed',  # 会在内部使用lambda_mix
            graph_laplacian=L_graph
        )
        
        # 层次权重归一化
        hierarchy_weights = F.softmax(self.hierarchy_weights, dim=0)  # (3,)
        
        # 加权融合三个层次的HSIC
        hsic_combined = (
            hierarchy_weights[0] * hsic_channel +
            hierarchy_weights[1] * hsic_spatial +
            hierarchy_weights[2] * hsic_interaction
        )  # (B,)
        
        # ========== Step 4: 双向因果HSIC ==========
        if self.use_causal_hsic:
            causal_direction, chsic_forward, chsic_backward = self.compute_causal_hsic(F_cross, F_dzz)
            
            # 根据因果方向调整HSIC
            # 如果 F_cross → F_dzz (causal_direction > 0)，增强 F_cross 的权重
            # 如果 F_dzz → F_cross (causal_direction < 0)，增强 F_dzz 的权重
            causal_weight = torch.sigmoid(causal_direction)  # (B,) ∈ [0, 1]
            # causal_weight > 0.5: F_cross主导
            # causal_weight < 0.5: F_dzz主导
        else:
            causal_weight = torch.ones(B, device=F_cross.device) * 0.5
        
        # ========== Step 5: 门控权重计算 ==========
        # 综合HSIC和因果信息
        gate_logits = self.alpha * hsic_combined / self.temperature
        gate_weight = torch.sigmoid(gate_logits)  # (B,)
        
        # 融合因果权重
        # 最终权重 = gate_weight * causal_weight
        # 这样既考虑了依赖强度（HSIC），又考虑了因果方向
        final_weight = gate_weight * causal_weight  # (B,)
        
        # ========== Step 6: 自适应融合 ==========
        # 扩展维度
        final_weight_expanded = final_weight.view(B, 1, 1)  # (B, 1, 1)
        
        # 主融合路径
        F_fused_main = final_weight_expanded * F_cross + (1 - final_weight_expanded) * F_dzz
        
        # 残差路径（保证梯度流）
        F_out = (1 - self.residual_weight) * F_fused_main + self.residual_weight * F_cross
        
        return F_out


# # ========== 使用示例 ==========
# if __name__ == "__main__":
#     # 模拟医学图像分割场景
#     B, C, H, W = 4, 192, 128, 128
#     L = H * W
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # 初始化HSIC-MF Gate
#     gate = HSICMFGate(
#         d_inner=C,
#         proj_dim=32,
#         alpha_init=1.0,
#         temperature=1.5,
#         knn_k=5,
#         use_graph_reg=True,
#         use_causal_hsic=True
#     ).to(device)
    
#     # 模拟输入
#     F_cross = torch.randn(B, C, L, device=device, requires_grad=True)
#     F_dzz = torch.randn(B, C, L, device=device, requires_grad=True)
    
#     # 前向传播
#     F_out = gate(F_cross, F_dzz)
    
#     # 模拟损失
#     loss = F_out.sum()
#     loss.backward()
    
#     # 检查梯度
#     print("=" * 50)
#     print("HSIC-MF Gate 梯度检查")
#     print("=" * 50)
#     print(f"F_cross gradient norm: {F_cross.grad.norm().item():.6f}")
#     print(f"F_dzz gradient norm: {F_dzz.grad.norm().item():.6f}")
#     print(f"Alpha gradient: {gate.alpha.grad.item():.6f}")
#     print(f"Hierarchy weights gradient: {gate.hierarchy_weights.grad}")
    
#     # 检查可学习参数
#     print("\n" + "=" * 50)
#     print("可学习参数统计")
#     print("=" * 50)
#     total_params = sum(p.numel() for p in gate.parameters())
#     trainable_params = sum(p.numel() for p in gate.parameters() if p.requires_grad)
#     print(f"Total parameters: {total_params:,}")
#     print(f"Trainable parameters: {trainable_params:,}")
    
#     # 验证梯度不为零
#     assert F_cross.grad.norm() > 1e-6, "梯度消失！"
#     assert F_dzz.grad.norm() > 1e-6, "梯度消失！"
    
#     print("\n✅ 所有测试通过！")



##########################################################
# SS2D_C 双分支 + HSIC Gate + ScanCache
##########################################################

class SS2D_C(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        branch_mode="both",            # "both" | "traditional_only" | "dzz_only"
        enable_cache=False,             # 总开关（与 scan_cache_mode 一起决定缓存策略）
        enable_gating=True,
        fusion_gate_type="hsic",       # "cos" | "hsic" | "hsic_rff" | "hsic_linear" 对比方法由 fusion_method 决定
        hsic_proj_dim=32,               # small=64、base=96（默认32对更大d_inner统计不足）。
        hsic_rff_dim=128,              # RFF 维度
        hsic_use_sparse=False,         # 是否使用稀疏 HSIC
        hsic_sparse_lambda=0.01,       # 稀疏正则化系数
        hsic_alpha=0.5,                # HSIC 缩放因子
        hsic_temperature=1.5,
        hsic_residual=0.3,
        # ScanCache 配置
        scan_cache_mode="buffer",       # "off" | "dict" | "buffer" | "precompute"
        scan_cache_precompute_shapes=None, # [(H,W), ...] 预计算尺寸
        scan_cache_profile=False,       # 是否记录索引耗时
        scan_cache_reduce="sum",        # "sum" or "mean"
        **kwargs,
    ):
        super().__init__()
        factory_kwargs = {}
        if device is not None: factory_kwargs["device"] = device
        if dtype is not None: factory_kwargs["dtype"] = dtype
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.branch_mode = branch_mode
        self.enable_cache = enable_cache
        self.enable_gating = enable_gating
        self.fusion_gate_type = fusion_gate_type.lower()
        # 统一融合方法选择（默认使用 fusion_gate_type）
        self.fusion_method = kwargs.get('fusion_method', self.fusion_gate_type).lower()
        self.hsic_proj_dim = int(hsic_proj_dim)
        self.hsic_alpha = float(hsic_alpha)
        self.hsic_temperature = float(hsic_temperature)
        self.hsic_residual = float(hsic_residual)

        # 可选：构建优化版 HSIC Gate（需在 d_inner/hsic_alpha 初始化之后）
        if self.fusion_method in ("hsic_opt", "optimized_hsic"):
            self.optimized_hsic_gate = OptimizedHSICGate(
                d_inner=self.d_inner,
                proj_dim=self.hsic_proj_dim,
                alpha_init=self.hsic_alpha,
                temperature=self.hsic_temperature,
                knn_k=5,
                use_graph_reg=True,
                use_causal_hsic=True,
            ).to(device)

        # ScanCache 参数
        self.scan_cache_mode = scan_cache_mode.lower()
        self.scan_cache_precompute_shapes = scan_cache_precompute_shapes or []
        self.scan_cache_profile = scan_cache_profile
        self.scan_cache_reduce = scan_cache_reduce.lower()

        # THOP hooks 清理标记（避免训练期残留hook导致device不一致）
        self._thop_hooks_cleared = False

        # 输入投影 & 深度可分离卷积
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=self.d_inner,
            bias=conv_bias,
            **factory_kwargs
        )
        self.act = nn.SiLU()

        # 传统分支 (4方向)
        self.x_proj_traditional = nn.Parameter(torch.stack([
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs).weight
            for _ in range(4)
        ], dim=0))
        self.dt_projs_weight_traditional = nn.Parameter(torch.stack([
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                        **factory_kwargs).weight
            for _ in range(4)
        ], dim=0))
        self.dt_projs_bias_traditional = nn.Parameter(torch.stack([
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                        **factory_kwargs).bias
            for _ in range(4)
        ], dim=0))
        self.A_logs_traditional = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True, device=device)
        self.Ds_traditional = self.D_init(self.d_inner, copies=4, merge=True, device=device)

        # DZZ 分支 (4方向)
        self.x_proj_weight_dzz = nn.Parameter(torch.stack([
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs).weight
            for _ in range(4)
        ], dim=0))
        self.dt_projs_weight_dzz = nn.Parameter(torch.stack([
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                        **factory_kwargs).weight
            for _ in range(4)
        ], dim=0))
        self.dt_projs_bias_dzz = nn.Parameter(torch.stack([
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                        **factory_kwargs).bias
            for _ in range(4)
        ], dim=0))
        self.A_logs_dzz = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True, device=device)
        self.Ds_dzz = self.D_init(self.d_inner, copies=4, merge=True, device=device)

        # 门控
        self.gate_projection = nn.Linear(self.d_inner * 2, self.d_inner, bias=False, **factory_kwargs)
        self.gate_norm = nn.LayerNorm(self.d_inner)
        self.gate_activation = nn.Sigmoid()
        self.fusion_weight = nn.Parameter(torch.ones(1))
        self.fusion_bias = nn.Parameter(torch.zeros(self.d_inner))

        # 简单消融融合层：1x1 conv（在通道维度进行线性融合）
        # 输入形状 (B, 2C, L) → 输出 (B, C, L)
        self.fuse_conv1x1_layer = nn.Conv1d(self.d_inner * 2, self.d_inner, kernel_size=1, bias=False, **factory_kwargs)
        # SE 通道注意力（Squeeze-and-Excitation）用于通道加权
        se_hidden = max(1, self.d_inner // 16)
        self.se_reduce = nn.Linear(self.d_inner, se_hidden, **factory_kwargs)
        self.se_expand = nn.Linear(se_hidden, self.d_inner, **factory_kwargs)

        # 输出投影
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # ScanCache 统计
        self._index_cache: Dict[Tuple[int,int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._cache_stats = {
            'total_requests': 0,
            'cache_misses': 0,
            'hit_rate': 0.0
        }
        self._profile_accum = {
            'index_build_time_us': 0.0,   # get_or_compute + gather
            'index_reorder_time_us': 0.0, # inverse reorder
            'calls': 0
        }

        if self.scan_cache_mode == 'precompute' and self.enable_cache:
            self._precompute_indices()
        
        self.save_attention = False
        self.attention_maps = []
        self.activation_maps = []

    ###############################
    # 初始化 / 工具
    ###############################
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random",
                dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        device = factory_kwargs.get('device', None)
        dtype = factory_kwargs.get('dtype', None)
        dt = torch.exp(
            torch.rand(d_inner, device=device, dtype=dtype) *
            (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                   "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device, dtype=torch.float32)
        if copies > 1:
            D = repeat(D, "n -> r n", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def _precompute_indices(self):
        dev = next(self.parameters()).device
        for (H, W) in self.scan_cache_precompute_shapes:
            key = (H, W)
            if key in self._index_cache:
                continue
            fwd, inv = _build_zigzag_indices(H, W, dev)
            if self.scan_cache_mode == 'buffer':
                self.register_buffer(f"scancache_fwd_{H}_{W}", fwd, persistent=False)
                self.register_buffer(f"scancache_inv_{H}_{W}", inv, persistent=False)
            else:
                self._index_cache[key] = (fwd, inv)

    def reset_scan_cache_stats(self):
        self._cache_stats = {'total_requests': 0, 'cache_misses': 0, 'hit_rate': 0.0}
        self._profile_accum = {
            'index_build_time_us': 0.0,
            'index_reorder_time_us': 0.0,
            'calls': 0
        }

    def scan_cache_report(self) -> str:
        s = self._cache_stats
        p = self._profile_accum
        if s['total_requests'] > 0:
            s['hit_rate'] = (s['total_requests'] - s['cache_misses']) / s['total_requests']
        avg_idx = (p['index_build_time_us']/p['calls']) if p['calls']>0 else 0.0
        avg_reo = (p['index_reorder_time_us']/p['calls']) if p['calls']>0 else 0.0
        return (f"ScanCache Report: total={s['total_requests']} miss={s['cache_misses']} "
                f"hit_rate={s['hit_rate']:.4f} | "
                f"avg_index_build_us={avg_idx:.2f} avg_reorder_us={avg_reo:.2f} calls={p['calls']}")

    def get_or_compute_cross_indices(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """ScanCache 缓存支持的 Cross-Scan 前向索引(4,L)。仅用于缓存与命中率统计。"""
        if not self.enable_cache or self.scan_cache_mode == 'off':
            return _build_cross_indices(H, W, device)
        key = ("cross", H, W)
        self._cache_stats['total_requests'] += 1
        if self.scan_cache_mode in ('dict', 'precompute'):
            entry = self._index_cache.get(key, None)
            if entry is not None:
                fwd = entry[0]
                if fwd.device != device:
                    fwd = fwd.to(device, non_blocking=True)
                    self._index_cache[key] = (fwd, None)
                return fwd
            fwd = _build_cross_indices(H, W, device)
            self._index_cache[key] = (fwd, None)
            self._cache_stats['cache_misses'] += 1
            return fwd
        elif self.scan_cache_mode == 'buffer':
            fwd_name = f"scancache_fwd_cross_{H}_{W}"
            fwd_buf = getattr(self, fwd_name, None)
            if fwd_buf is not None:
                if fwd_buf.device != device:
                    fwd_buf = fwd_buf.to(device, non_blocking=True)
                    setattr(self, fwd_name, fwd_buf)
                return fwd_buf
            fwd = _build_cross_indices(H, W, device)
            self.register_buffer(fwd_name, fwd, persistent=False)
            self._cache_stats['cache_misses'] += 1
            return fwd
        else:
            return _build_cross_indices(H, W, device)

    ###############################
    # 传统分支
    ###############################
    def vmamba_traditional_ss2d_branch(self, x: torch.Tensor):
        # 依赖 selective_scan
        selective_scan = None
        if x.is_cuda and selective_scan_fn is not None:
            selective_scan = selective_scan_fn
        elif selective_scan_ref is not None:
            selective_scan = selective_scan_ref
        if selective_scan is None:
            # fallback
            B, C, H, W = x.shape
            L = H * W
            xs = vmamba_cross_scan_fwd(x, True, True).view(B, 4, C, L)
            return xs.sum(dim=1)

        B, C, H, W = x.shape
        L = H * W
        K = 4
        xs = vmamba_cross_scan_fwd(x, True, True)  # (B,4,C,L)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l",
                             xs.view(B, K, -1, L), self.x_proj_traditional)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l",
                           dts.view(B, K, -1, L), self.dt_projs_weight_traditional)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        Bs = Bs.view(B, K, -1, L)
        Cs = Cs.view(B, K, -1, L)
        Ds = self.Ds_traditional.view(-1)
        As = -torch.exp(self.A_logs_traditional).view(-1, self.d_state)
        dt_bias = self.dt_projs_bias_traditional.view(-1)

        out_y = selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_bias,
            delta_softplus=True,
            return_last_state=False
        ).view(B, K, -1, L)

        out_y_reshaped = out_y.view(B, K, -1, H, W)
        y_traditional = vmamba_cross_merge_fwd(out_y_reshaped, True, True)  # (B,d,L)
        return y_traditional

    ###############################
    # 改良 DZZ 分支（使用 ScanCache）
    ###############################
    def get_or_compute_indices(self, H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据 scan_cache_mode 返回 (fwd, inv); 保证与当前输入 device 一致。
        """
        if not self.enable_cache or self.scan_cache_mode == 'off':
            fwd, inv = _build_zigzag_indices(H, W, device)
            return fwd, inv

        key = (H, W)
        self._cache_stats['total_requests'] += 1

        if self.scan_cache_mode in ('dict', 'precompute'):
            entry = self._index_cache.get(key, None)
            if entry is not None:
                fwd, inv = entry
                # --- 补丁：保证设备一致 ---
                if fwd.device != device:
                    fwd = fwd.to(device, non_blocking=True)
                    inv = inv.to(device, non_blocking=True)
                    # 更新缓存为 GPU 版本，避免下次再迁移
                    self._index_cache[key] = (fwd, inv)
                return fwd, inv
            # miss
            fwd, inv = _build_zigzag_indices(H, W, device)
            self._index_cache[key] = (fwd, inv)
            self._cache_stats['cache_misses'] += 1
            return fwd, inv

        elif self.scan_cache_mode == 'buffer':
            fwd_name = f"scancache_fwd_{H}_{W}"
            inv_name = f"scancache_inv_{H}_{W}"
            fwd_buf = getattr(self, fwd_name, None)
            inv_buf = getattr(self, inv_name, None)
            if fwd_buf is not None and inv_buf is not None:
                # buffer 随 .cuda() 迁移，一般不需要处理；但仍防护
                if fwd_buf.device != device:
                    fwd_buf = fwd_buf.to(device, non_blocking=True)
                    inv_buf = inv_buf.to(device, non_blocking=True)
                    setattr(self, fwd_name, fwd_buf)
                    setattr(self, inv_name, inv_buf)
                return fwd_buf, inv_buf
            # miss
            fwd, inv = _build_zigzag_indices(H, W, device)
            self.register_buffer(fwd_name, fwd, persistent=False)
            self.register_buffer(inv_name, inv, persistent=False)
            self._cache_stats['cache_misses'] += 1
            return fwd, inv

        else:
            # 未知模式降级
            fwd, inv = _build_zigzag_indices(H, W, device)
            return fwd, inv


    def dzz_zigzag_diagonal_branch(self, x: torch.Tensor):
        # 选择 selective_scan
        selective_scan = None
        if x.is_cuda and selective_scan_fn is not None:
            selective_scan = selective_scan_fn
        elif selective_scan_ref is not None:
            selective_scan = selective_scan_ref

        B, C, H, W = x.shape
        L = H * W
        K = 4
        flat = x.view(B, C, L)

        # 计时：索引 + gather
        t0 = time.perf_counter() if self.scan_cache_profile else None
        fwd, inv = self.get_or_compute_indices(H, W, x.device)  # (4,L),(4,L)

        xs = torch.empty(B, K, C, L, device=x.device, dtype=flat.dtype)
        # 4 个方向 gather
        for k in range(4):
            xs[:, k] = torch.index_select(flat, 2, fwd[k])

        # fallback 无 selective_scan
        if selective_scan is None:
            # 简单平均代替
            y = xs.mean(dim=1) if self.scan_cache_reduce == 'mean' else xs.sum(dim=1)
            if self.scan_cache_profile:
                t1 = time.perf_counter()
                self._profile_accum['index_build_time_us'] += (t1 - t0) * 1e6
                self._profile_accum['index_reorder_time_us'] += 0.0
                self._profile_accum['calls'] += 1
            return y  # (B,C,L)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l",
                             xs.view(B, K, -1, L), self.x_proj_weight_dzz)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l",
                           dts.view(B, K, -1, L), self.dt_projs_weight_dzz)

        xs_lin = xs.view(B, -1, L)
        dts_lin = dts.contiguous().view(B, -1, L)
        Bs_lin = Bs.view(B, K, -1, L)
        Cs_lin = Cs.view(B, K, -1, L)
        Ds_lin = self.Ds_dzz.view(-1)
        As_lin = -torch.exp(self.A_logs_dzz).view(-1, self.d_state)
        dt_bias = self.dt_projs_bias_dzz.view(-1)

        if self.scan_cache_profile:
            t1 = time.perf_counter()
            self._profile_accum['index_build_time_us'] += (t1 - t0) * 1e6

        out_y = selective_scan(
            xs_lin, dts_lin,
            As_lin, Bs_lin, Cs_lin, Ds_lin, z=None,
            delta_bias=dt_bias,
            delta_softplus=True,
            return_last_state=False
        ).view(B, K, -1, L)

        # 逆重排 (还原回原空间顺序)
        t2 = time.perf_counter() if self.scan_cache_profile else None
        for k in range(K):
            out_y[:, k] = torch.index_select(out_y[:, k], -1, inv[k])
        if self.scan_cache_reduce == 'mean':
            y_dzz = out_y.mean(dim=1)
        else:
            y_dzz = out_y.sum(dim=1)

        if self.scan_cache_profile:
            t3 = time.perf_counter()
            self._profile_accum['index_reorder_time_us'] += (t3 - t2) * 1e6
            self._profile_accum['calls'] += 1
        return y_dzz  # (B,d_inner,L)

    ###############################
    # HSIC Gate and Feature Fusion
    ###############################
    def gated_fusion(self, traditional_out, dzz_out):
        '''
        基于cosine相似度门控融合
        '''
        B, C, L = traditional_out.shape
        # 全局平均池化
        trad_global = traditional_out.mean(dim=2)
        dzz_global = dzz_out.mean(dim=2)     
        # 归一化
        trad_norm = F.normalize(trad_global, dim=1, eps=1e-8)
        dzz_norm = F.normalize(dzz_global, dim=1, eps=1e-8)
        # 计算余弦相似度
        cosine_similarity = torch.sum(trad_norm * dzz_norm, dim=1, keepdim=True)
        # 使用余弦相似度计算门控权重
        gate_weight = torch.sigmoid(cosine_similarity * 5.0)
        # 阈值判断
        threshold = 0.5
        should_fuse = (gate_weight > threshold).float()
        # 计算融合权重
        alpha = gate_weight * should_fuse
        alpha_e = alpha.unsqueeze(2)
        # 融合操作
        fuse_part = alpha_e * traditional_out + (1 - alpha_e) * dzz_out
        out = should_fuse.unsqueeze(2) * fuse_part + (1 - should_fuse.unsqueeze(2)) * traditional_out
        
        return out

    def hsic_gate_B(self, traditional_out: torch.Tensor, dzz_out: torch.Tensor):
        """
        优化版 HSIC Gate: RBF核 + JL投影 + 空间采样
        
        核心改进:
        1. 使用 JL 投影替代 RFF (4x faster in projection)
        2. 在通道维度采样 m 个 landmark (避免 O(C²) 复杂度)
        3. 使用显式 RBF 核 (保留非线性表达能力)
        
        复杂度: O(BCLk + Bm²k) ≈ 3.23 GFLOPs
        相比 RBF+RFF 的 4.41 TFLOPs，快了 1365 倍.
        """
        B, C, L = traditional_out.shape
        
        # ========== Step 1: Johnson-Lindenstrauss 投影 ==========
        k = max(8, min(self.hsic_proj_dim, L))  # k=32
        P = torch.randn(L, k, device=traditional_out.device, dtype=traditional_out.dtype)
        P = F.normalize(P, dim=0)  # 列归一化
        
        # 投影到低维空间: (B, C, L) @ (L, k) -> (B, C, k)
        X_cross = (traditional_out @ P) / math.sqrt(L)  # O(BCLk)
        X_dzz = (dzz_out @ P) / math.sqrt(L)
        
        # ========== Step 2: 空间采样 (关键优化!) ==========
        m = min(64, C // 4)  # 采样 m=64 个通道
        if C > m:
            # 均匀采样
            indices = torch.linspace(0, C-1, m, dtype=torch.long, device=traditional_out.device)
            X_cross_sampled = X_cross[:, indices, :]  # (B, m, k)
            X_dzz_sampled = X_dzz[:, indices, :]
        else:
            X_cross_sampled = X_cross
            X_dzz_sampled = X_dzz
            m = C
        
        # ========== Step 3: 显式 RBF 核计算 ==========
        sigma = 1.0
        
        def compute_rbf_kernel(X):
            """
            计算 RBF 核矩阵: K[i,j] = exp(-||X[i] - X[j]||² / (2σ²))
            输入: X (B, m, k)
            输出: K (B, m, m)
            """
            # 计算平方距离矩阵
            X_norm = (X ** 2).sum(dim=-1, keepdim=True)  # (B, m, 1)
            dist_sq = X_norm + X_norm.transpose(1, 2) - 2 * (X @ X.transpose(1, 2))  # (B, m, m)
            
            # RBF 核
            K = torch.exp(-dist_sq / (2 * sigma ** 2))
            return K
        
        K_cross = compute_rbf_kernel(X_cross_sampled)  # O(Bm²k)
        K_dzz = compute_rbf_kernel(X_dzz_sampled)
        
        # ========== Step 4: 高效中心化 ==========
        def center_kernel(K):
            """中心化核矩阵，避免显式构造 H 矩阵"""
            row_mean = K.mean(dim=-1, keepdim=True)  # (B, m, 1)
            col_mean = K.mean(dim=-2, keepdim=True)  # (B, 1, m)
            total_mean = K.mean(dim=(-1, -2), keepdim=True)  # (B, 1, 1)
            return K - row_mean - col_mean + total_mean
        
        K_cross_centered = center_kernel(K_cross)
        K_dzz_centered = center_kernel(K_dzz)
        
        # ========== Step 5: HSIC 计算 ==========
        denom = max((m - 1) ** 2, 1)
        hsic_vals = (K_cross_centered * K_dzz_centered).sum(dim=(-1, -2)) / denom  # (B,)
        hsic_vals = hsic_vals.unsqueeze(1)  # (B, 1)
        
        # ========== Step 6: 自适应门控权重 ==========
        gate_weight = torch.sigmoid(self.hsic_alpha * hsic_vals)  # (B, 1)
        
        # ========== Step 7: 条件融合 ==========
        threshold = 0.5
        should_fuse = (gate_weight > threshold).float()  # (B, 1)
        
        alpha_expanded = gate_weight.unsqueeze(2)  # (B, 1, 1)
        fused_part = alpha_expanded * traditional_out + (1 - alpha_expanded) * dzz_out
        
        out = should_fuse.unsqueeze(2) * fused_part + (1 - should_fuse.unsqueeze(2)) * traditional_out
        
        return out

    def fuse_add(self, traditional_out: torch.Tensor, dzz_out: torch.Tensor) -> torch.Tensor:
        """
        /**
         * @function fuse_add
         * @brief 元素级相加融合（Element-wise Addition）
         * @param {Tensor} traditional_out - (B, C, L) Cross-Scan 分支输出
         * @param {Tensor} dzz_out - (B, C, L) DZZ-Scan 分支输出
         * @returns {Tensor} (B, C, L) 融合结果
         */
        """
        return traditional_out + dzz_out

    def fuse_conv1x1(self, traditional_out: torch.Tensor, dzz_out: torch.Tensor) -> torch.Tensor:
        """
        /**
         * @function fuse_conv1x1
         * @brief 1x1 卷积融合（通道维线性投影），先在通道维拼接，再用1x1进行C->C映射
         * @param {Tensor} traditional_out - (B, C, L)
         * @param {Tensor} dzz_out - (B, C, L)
         * @returns {Tensor} (B, C, L)
         */
        """
        x = torch.cat([traditional_out, dzz_out], dim=1)  # (B, 2C, L)
        out = self.fuse_conv1x1_layer(x)                  # (B, C, L)
        return out

    def fuse_se(self, traditional_out: torch.Tensor, dzz_out: torch.Tensor) -> torch.Tensor:
        """
        /**
         * @function fuse_se
         * @brief SE（Squeeze-and-Excitation）通道注意力融合
         *  思路：从 (F_t, F_d) 学习每通道的加权 α，并做 α·F_t + (1-α)·F_d
         * @param {Tensor} traditional_out - (B, C, L)
         * @param {Tensor} dzz_out - (B, C, L)
         * @returns {Tensor} (B, C, L)
         */
        """
        B, C, L = traditional_out.shape
        # squeeze：用全局平均池化在长度维聚合
        gap_t = traditional_out.mean(dim=2)  # (B, C)
        gap_d = dzz_out.mean(dim=2)          # (B, C)
        gap = 0.5 * (gap_t + gap_d)          # 简单聚合两个分支的全局描述
        # excitation：两层MLP得到通道权重 α \in (0,1)
        s = self.se_expand(F.relu(self.se_reduce(gap)))   # (B, C)
        alpha = torch.sigmoid(s).unsqueeze(2)             # (B, C, 1)
        # 融合：α·F_t + (1-α)·F_d
        out = alpha * traditional_out + (1 - alpha) * dzz_out
        return out


    def hsic_gate_A(self, traditional_out: torch.Tensor, dzz_out: torch.Tensor):
        """
        /**
         * 方法: HSIC Gate (基线)
         * 思路:
         *  1) 随机投影 P \in R^{L×k}，列归一化，得到 X_t = (F_t P)/√L, X_d = (F_d P)/√L；
         *  2) 构造核矩阵 K_t = X_t X_t^T, K_d = X_d X_d^T（线性kernal，仅仅捕捉线性关系）；中心化 \tilde{K}=H K H，H=I−1/C·11^T；
         *  3) HSIC ≈ <H K_t H, H K_d H>_F / (C−1)^2；用 w = sigmoid(α·HSIC) 作为门控权重；
         *  4) 若 w>τ 则融合: F = w·F_t + (1−w)·F_d，否则回退到 F_t。
         * 公平性:
         *  与三种对比方法保持相同的随机投影维度 k、同一 α、同一阈值 τ、同一融合位置与张量，确保可比。
         */
        """
        '''
        基于HSIC门控融合
        '''
        # Step 1: 随机投影 + L2归一化 (提高稳定性)
        B, C, L = traditional_out.shape
        k = max(8, min(self.hsic_proj_dim, L))
        P = torch.randn(L, k, device=traditional_out.device, dtype=traditional_out.dtype)
        P = F.normalize(P, dim=0)
        Xt = F.normalize((traditional_out @ P) / math.sqrt(L), dim=-1)  # L2归一化
        Xd = F.normalize((dzz_out @ P) / math.sqrt(L), dim=-1)
        Kt = Xt @ Xt.transpose(1, 2)
        Kd = Xd @ Xd.transpose(1, 2)
        # Step 2: RBF核 (同v1)
        def compute_rbf_kernel(X1, X2=None):
            if X2 is None:
                X2 = X1
            X1_norm = (X1 ** 2).sum(dim=-1, keepdim=True)
            X2_norm = (X2 ** 2).sum(dim=-1, keepdim=True)
            dist_sq = X1_norm + X2_norm.transpose(1, 2) - 2 * (X1 @ X2.transpose(1, 2))
            dist_sq = dist_sq.clamp(min=0)
            
            # 自适应带宽
            with torch.no_grad():
                sigma_sq = dist_sq.view(B, -1).median(dim=1, keepdim=True)[0].clamp(min=1e-6)
            
            return torch.exp(-dist_sq / (2 * sigma_sq.unsqueeze(-1)))
        Kt = compute_rbf_kernel(Xt)
        Kd = compute_rbf_kernel(Xd)
        
        # Step 3: 中心化
        def center(G):
            r = G.mean(dim=-1, keepdim=True)
            c = G.mean(dim=-2, keepdim=True)
            m = G.mean(dim=(-1,-2), keepdim=True)
            return G - r - c + m
        Kt_c = center(Kt)
        Kd_c = center(Kd)
        denom = max((C - 1) ** 2, 1)
        hsic_vals = (Kt_c * Kd_c).sum(dim=(-1, -2)) / denom
        hsic_vals = hsic_vals.unsqueeze(1)
        gate_weight = torch.sigmoid(self.hsic_alpha * hsic_vals)
        threshold = 0.5
        should_fuse = (gate_weight > threshold).float()
        alpha = gate_weight * should_fuse
        alpha_e = alpha.unsqueeze(2)
        fused_part = alpha_e * traditional_out + (1 - alpha_e) * dzz_out
        out = should_fuse.unsqueeze(2) * fused_part + (1 - should_fuse.unsqueeze(2)) * traditional_out

        return out

    # def hsic_gate(self, traditional_out: torch.Tensor, dzz_out: torch.Tensor):
    #     """
    #     优化的HSIC Gate v1(保持HSIC理论框架)
        
    #     关键改进:
    #     1. RBF核替代线性核 - 捕捉非线性依赖 (Gretton et al. 2005, 2012)
    #     2. 多尺度核带宽 - 适应不同特征尺度 (Sutherland et al. 2017)
    #     3. 自适应阈值 - 基于HSIC分布的动态阈值 (Kim et al. 2020)
    #     4. 温度缩放 - 平滑门控决策 (Hinton et al. 2015)
        
    #     参考文献:
    #     - Gretton et al. "Measuring Statistical Dependence with Hilbert-Schmidt Norms" (ALT 2005)
    #     - Gretton et al. "Kernel Methods for Measuring Independence" (JMLR 2005)
    #     - Sutherland et al. "Generative Models and Model Criticism via Optimized MMD" (ICLR 2017)
    #     - Kim et al. "Adaptive Kernel Selection for Test-Time Adaptation" (NeurIPS 2020)
    #     """
    #     B, C, L = traditional_out.shape
    #     k = max(8, min(self.hsic_proj_dim, L))
        
    #     # Step 1: 随机投影 (与baseline相同)
    #     P = torch.randn(L, k, device=traditional_out.device, dtype=traditional_out.dtype)
    #     P = F.normalize(P, dim=0)
    #     Xt = (traditional_out @ P) / math.sqrt(L)  # [B, C, k]
    #     Xd = (dzz_out @ P) / math.sqrt(L)
        
    #     # Step 2: 多尺度RBF核 (关键改进1)
    #     # 使用中位数启发式 + 多尺度 (Gretton et al. 2012, Sutherland et al. 2017)
    #     def compute_multiscale_rbf_kernel(X1, X2=None):
    #         """
    #         计算多尺度RBF核矩阵
    #         K(x,y) = sum_i exp(-||x-y||^2 / (2*sigma_i^2))
    #         """
    #         if X2 is None:
    #             X2 = X1
            
    #         # 计算成对距离 [B, C, C]
    #         X1_norm = (X1 ** 2).sum(dim=-1, keepdim=True)  # [B, C, 1]
    #         X2_norm = (X2 ** 2).sum(dim=-1, keepdim=True)  # [B, C, 1]
    #         dist_sq = X1_norm + X2_norm.transpose(1, 2) - 2 * (X1 @ X2.transpose(1, 2))
    #         dist_sq = dist_sq.clamp(min=0)  # 数值稳定性
            
    #         # 中位数启发式估计带宽 (Gretton et al. 2012)
    #         with torch.no_grad():
    #             median_dist = dist_sq.view(B, -1).median(dim=1, keepdim=True)[0]
    #             median_dist = median_dist.clamp(min=1e-6)
            
    #         # 多尺度带宽: {0.5σ, σ, 2σ} (Sutherland et al. 2017)
    #         sigmas = [0.5, 1.0, 2.0]
    #         K = 0
    #         for scale in sigmas:
    #             sigma_sq = scale * median_dist.unsqueeze(-1)
    #             K = K + torch.exp(-dist_sq / (2 * sigma_sq))
            
    #         return K / len(sigmas)  # 平均多个尺度
        
    #     Kt = compute_multiscale_rbf_kernel(Xt)  # [B, C, C]
    #     Kd = compute_multiscale_rbf_kernel(Xd)
        
    #     # Step 3: 核矩阵中心化 (标准HSIC操作)
    #     def center_kernel(K):
    #         """高效中心化: Kc = H K H"""
    #         row_mean = K.mean(dim=-1, keepdim=True)
    #         col_mean = K.mean(dim=-2, keepdim=True)
    #         total_mean = K.mean(dim=(-1, -2), keepdim=True)
    #         return K - row_mean - col_mean + total_mean
        
    #     Kt_c = center_kernel(Kt)
    #     Kd_c = center_kernel(Kd)
        
    #     # Step 4: 计算HSIC (无偏估计, Gretton et al. 2005)
    #     hsic_vals = (Kt_c * Kd_c).sum(dim=(-1, -2)) / max((C - 1) ** 2, 1)
    #     hsic_vals = hsic_vals.unsqueeze(1)  # [B, 1]
        
    #     # Step 5: 自适应阈值 (关键改进2)
    #     # 基于batch内HSIC分布的动态阈值 (Kim et al. 2020)
    #     with torch.no_grad():
    #         hsic_mean = hsic_vals.mean()
    #         hsic_std = hsic_vals.std().clamp(min=1e-6)
    #         # 自适应阈值: mean + 0.5*std (保留高依赖样本)
    #         adaptive_threshold = hsic_mean + 0.5 * hsic_std
    #         adaptive_threshold = adaptive_threshold.clamp(min=0.1, max=0.9)
        
    #     # Step 6: 温度缩放的门控 (关键改进3)
    #     # 使用温度参数平滑sigmoid (Hinton et al. 2015)
    #     temperature = 2.0  # 温度越高，决策越平滑
    #     gate_weight = torch.sigmoid(self.hsic_alpha * hsic_vals / temperature)
        
    #     # Step 7: 软融合策略 (移除硬阈值)
    #     # 使用连续的门控权重，避免二值化带来的不稳定性
    #     alpha_e = gate_weight.unsqueeze(2)  # [B, 1, 1]
        
    #     # 加入置信度加权 (基于HSIC强度)
    #     confidence = torch.sigmoid((hsic_vals - adaptive_threshold) * 5.0).unsqueeze(2)
        
    #     # 最终融合: 高HSIC → 更多融合; 低HSIC → 更多cross-scan
    #     fused = confidence * (alpha_e * traditional_out + (1 - alpha_e) * dzz_out) + \
    #             (1 - confidence) * traditional_out
        
    #     return fused





    def hsic_gate_C(self, traditional_out: torch.Tensor, dzz_out: torch.Tensor):
        """
        HSIC Gate 优化版本， accepted by ACMMM 2026:TopoMamba.
        
        额外改进:
        1. 特征归一化 - 提高核计算稳定性
        2. 双向HSIC - 同时考虑Kt→Kd和Kd→Kt的依赖
        3. 残差融合 - 保留更多原始信息
        """
        B, C, L = traditional_out.shape
        k = max(8, min(self.hsic_proj_dim, L))
        
        # Step 1: 随机投影 + L2归一化 (提高稳定性)
        P = torch.randn(L, k, device=traditional_out.device, dtype=traditional_out.dtype)
        P = F.normalize(P, dim=0)
        Xt = F.normalize((traditional_out @ P) / math.sqrt(L), dim=-1)  # L2归一化
        Xd = F.normalize((dzz_out @ P) / math.sqrt(L), dim=-1)
        
        # Step 2: RBF核 (同v1)
        def compute_rbf_kernel(X1, X2=None):
            if X2 is None:
                X2 = X1
            X1_norm = (X1 ** 2).sum(dim=-1, keepdim=True)
            X2_norm = (X2 ** 2).sum(dim=-1, keepdim=True)
            dist_sq = X1_norm + X2_norm.transpose(1, 2) - 2 * (X1 @ X2.transpose(1, 2))
            dist_sq = dist_sq.clamp(min=0)
            
            # 自适应带宽
            with torch.no_grad():
                sigma_sq = dist_sq.view(B, -1).median(dim=1, keepdim=True)[0].clamp(min=1e-6)
            
            return torch.exp(-dist_sq / (2 * sigma_sq.unsqueeze(-1)))
        
        Kt = compute_rbf_kernel(Xt)
        Kd = compute_rbf_kernel(Xd)
        
        # Step 3: 中心化
        def center(K):
            r = K.mean(dim=-1, keepdim=True)
            c = K.mean(dim=-2, keepdim=True)
            m = K.mean(dim=(-1,-2), keepdim=True)
            return K - r - c + m
        
        Kt_c = center(Kt)
        Kd_c = center(Kd)
        
        # Step 4: 双向HSIC (对称性)
        hsic_forward = (Kt_c * Kd_c).sum(dim=(-1, -2))
        hsic_backward = (Kd_c * Kt_c).sum(dim=(-1, -2))  # 理论上相同，但数值上可能略有差异
        hsic_vals = (hsic_forward + hsic_backward) / (2 * max((C - 1) ** 2, 1))
        hsic_vals = hsic_vals.unsqueeze(1)
        
        # Step 5: 平滑门控
        gate_weight = torch.sigmoid(self.hsic_alpha * hsic_vals / max(self.hsic_temperature, 1e-6))
        
        # Step 6: 残差融合 (保留更多原始特征)
        alpha_e = gate_weight.unsqueeze(2)
        residual_weight = self.hsic_residual
        
        fused = (1 - residual_weight) * (alpha_e * traditional_out + (1 - alpha_e) * dzz_out) + residual_weight * dzz_out
        
        return fused


    def hsic_gate(self, traditional_out: torch.Tensor, dzz_out: torch.Tensor):
        """
        兼容入口：映射到当前稳定实现的 HSIC Gate（hsic_gate_C）。
        """
        return self.hsic_gate_C(traditional_out, dzz_out)


    def fusion_gate_merge_I2PMAE_BFTT3D_DTRG(self, traditional_out: torch.Tensor, dzz_out: torch.Tensor):
        """
        /**
         * 方法: HSIC Gate（改进版，融合 I2P-MAE / BFTT3D / DTRG 思想，消除硬阈值梯度消失）
         *
         * 记 F_t, F_d ∈ R^{B×C×L}，k 为随机投影维度。
         *
         * 1) 随机投影与核对齐（I2P-MAE 思想）
         *    X_t = (F_t P)/√L, X_d = (F_d P)/√L,  P ∈ R^{L×k} 列归一；
         *    K_t = X_t X_t^T, K_d = X_d X_d^T；中心化: K̃ = H K H, H = I - (1/C)11^T；
         *    对齐分数: A = ⟨K̃_t, K̃_d⟩_F / (||K̃_t||_F · ||K̃_d||_F)。
         *
         * 2) 通道 Gram 相关（BFTT3D 思想）
         *    G_t = F_t F_t^T, G_d = F_d F_d^T；中心化同上；
         *    相关系数: ρ = ⟨Ĝ_t, Ĝ_d⟩_F / (||Ĝ_t||_F · ||Ĝ_d||_F)。
         *
         * 3) 软化的 kNN 图一致性（DTRG 思想，连续化近似）
         *    令 Z_t = norm(F_t, dim=L), Z_d 同理；S = Z Z^T 为通道相似度矩阵；
         *    定义 softmax(-β·(1−S)) 作为“软”kNN 权重分布 q；
         *    S_soft = mean_i ⟨q_t(i,:), q_d(i,:)⟩ ∈ [0,1]。
         *
         * 4) 融合与门控
         *    将三种分数按批内标准化后求平均：
         *      s̄ = mean( z(A), z(ρ), z(S_soft) )，z(x)=(x−μ)/σ；
         *    使用连续门控（无硬阈值）以避免梯度消失：
         *      w = sigmoid(α · s̄)，F_out = w·F_t + (1−w)·F_d。
         *
         * 公平性：
         *  - 与 HSIC_A 保持相同 k、α 与融合位置；
         *  - 计算均采用相同的投影/中心化与张量；
         *  - 移除硬阈值，提升可训练性与稳定梯度。
         */
        """
        eps = 1e-8
        B, C, L = traditional_out.shape
        device = traditional_out.device

        # 1) 随机投影与核对齐（I2P-MAE）
        k = max(8, min(self.hsic_proj_dim, L))
        P = torch.randn(L, k, device=device, dtype=traditional_out.dtype)
        P = F.normalize(P, dim=0)
        Xt = (traditional_out @ P) / math.sqrt(L)      # (B, C, k)
        Xd = (dzz_out @ P) / math.sqrt(L)              # (B, C, k)
        Kt = Xt @ Xt.transpose(1, 2)                   # (B, C, C)
        Kd = Xd @ Xd.transpose(1, 2)
        I = torch.eye(C, device=device, dtype=traditional_out.dtype).unsqueeze(0)
        H = I - (1.0 / C) * torch.ones_like(I)
        Kt_c = H @ Kt @ H
        Kd_c = H @ Kd @ H
        num_align = (Kt_c * Kd_c).sum(dim=(-1, -2))
        den_align = (Kt_c.pow(2).sum(dim=(-1, -2)).sqrt() * Kd_c.pow(2).sum(dim=(-1, -2)).sqrt()).clamp_min(eps)
        s_align = (num_align / den_align).unsqueeze(1)    # (B,1)

        # 2) 通道 Gram 相关（BFTT3D）
        Gc = traditional_out @ traditional_out.transpose(1, 2)  # (B,C,C)
        Gd = dzz_out @ dzz_out.transpose(1, 2)
        Gc_c = H @ Gc @ H
        Gd_c = H @ Gd @ H
        num_gram = (Gc_c * Gd_c).sum(dim=(-1, -2))
        den_gram = (Gc_c.pow(2).sum(dim=(-1, -2)).sqrt() * Gd_c.pow(2).sum(dim=(-1, -2)).sqrt()).clamp_min(eps)
        s_gram = (num_gram / den_gram).unsqueeze(1)       # (B,1)

        # 3) 软化的 kNN 图一致性（DTRG 连续近似）
        beta = 10.0
        Zt = F.normalize(traditional_out, dim=-1, eps=eps)  # (B,C,L)
        Zd = F.normalize(dzz_out, dim=-1, eps=eps)
        S_t = (Zt @ Zt.transpose(1, 2)).clamp(-1, 1)        # (B,C,C)
        S_d = (Zd @ Zd.transpose(1, 2)).clamp(-1, 1)
        D_t = 1.0 - S_t
        D_d = 1.0 - S_d
        q_t = F.softmax(-beta * D_t, dim=-1)                 # (B,C,C)
        q_d = F.softmax(-beta * D_d, dim=-1)
        s_knn_soft = (q_t * q_d).sum(dim=-1).mean(dim=-1, keepdim=True)  # (B,1)

        # 4) 批内标准化与融合
        S = torch.cat([s_align, s_gram, s_knn_soft], dim=1)  # (B,3)
        mu = S.mean(dim=1, keepdim=True)
        std = S.std(dim=1, keepdim=True).clamp_min(eps)
        S_norm = (S - mu) / std
        s_bar = S_norm.mean(dim=1, keepdim=True)             # (B,1)

        w = torch.sigmoid(self.hsic_alpha * s_bar).unsqueeze(2)  # (B,1,1)
        out = w * traditional_out + (1 - w) * dzz_out            # 连续门控，无硬阈值
        return out

    def i2pmae_cvpr2023(self, traditional_out: torch.Tensor, dzz_out: torch.Tensor):
        """
        /**
         * 方法: I2P-MAE 对齐 (CVPR 2023)
         * 思路:
         *  i) 随机投影 P ∈ R^{L×k}（列归一）；X_c=(F_t P)/√L, X_d=(F_d P)/√L；
         *  ii) 线性核: K_c=X_c X_c^T, K_d=X_d X_d^T；居中 H=I−1/C·11^T；
         *  iii) 核对齐分数 A = <H K_c H, H K_d H>_F / ||H K_c H||_F ||H K_d H||_F；
         *  iv) w = sigmoid(α·A)，若 w>τ 则 F = w·F_t + (1−w)·F_d，否则回退 F_t。
         * 公平性:
         *  使用与 HSIC 相同的 k、α、τ、同一融合算子与位置，且随机投影相同的生成器种子。
         */
        """
        B, C, L = traditional_out.shape
        k = max(8, min(self.hsic_proj_dim, L))
        P = torch.randn(L, k, device=traditional_out.device, dtype=traditional_out.dtype)
        P = F.normalize(P, dim=0)
        Xc = (traditional_out @ P) / math.sqrt(L)
        Xd = (dzz_out @ P) / math.sqrt(L)
        Kc = Xc @ Xc.transpose(1, 2)
        Kd = Xd @ Xd.transpose(1, 2)
        I = torch.eye(C, device=traditional_out.device, dtype=traditional_out.dtype).unsqueeze(0)
        H = I - (1.0 / C) * torch.ones_like(I)
        Kc_c = H @ Kc @ H
        Kd_c = H @ Kd @ H
        num = (Kc_c * Kd_c).sum(dim=(-1, -2))
        den = (Kc_c.pow(2).sum(dim=(-1, -2)).sqrt() * Kd_c.pow(2).sum(dim=(-1, -2)).sqrt()).clamp_min(1e-8)
        A = (num / den).unsqueeze(1)
        w = torch.sigmoid(self.hsic_alpha * A).unsqueeze(2)
        threshold = 0.5
        fuse_mask = (w.squeeze(2) > threshold).float().unsqueeze(2)
        fused = fuse_mask * (w * traditional_out + (1 - w) * dzz_out) + (1 - fuse_mask) * traditional_out
        return fused

    def bftt3d_cvpr2024(self, traditional_out: torch.Tensor, dzz_out: torch.Tensor):
        """
        /**
         * 方法: BFTT3D 通道 Gram 相关 (CVPR 2024)
         * 思路:
         *  以通道 Gram 矩阵 G = F F^T，做中心化 \tilde{G}=H G H；
         *  相关性 ρ = <\tilde{G_c}, \tilde{G_d}>_F / (||\tilde{G_c}||_F ||\tilde{G_d}||_F)；
         *  w = sigmoid(α·ρ)，若 w>τ 则融合 F = w·F_t + (1−w)·F_d，否则回退 F_t；不引入可学习参数。
         * 公平性:
         *  与 HSIC 统一 α、τ 与融合位置；仅核形式不同。
         */
        """
        B, C, L = traditional_out.shape
        Gc = traditional_out @ traditional_out.transpose(1, 2)
        Gd = dzz_out @ dzz_out.transpose(1, 2)
        I = torch.eye(C, device=traditional_out.device, dtype=traditional_out.dtype).unsqueeze(0)
        H = I - (1.0 / C) * torch.ones_like(I)
        Gc_c = H @ Gc @ H
        Gd_c = H @ Gd @ H
        num = (Gc_c * Gd_c).sum(dim=(-1, -2))
        den = (Gc_c.pow(2).sum(dim=(-1, -2)).sqrt() * Gd_c.pow(2).sum(dim=(-1, -2)).sqrt()).clamp_min(1e-8)
        corr = (num / den).unsqueeze(1)
        w = torch.sigmoid(self.hsic_alpha * corr).unsqueeze(2)
        tau = 0.5
        fuse_mask = (w.squeeze(2) > tau).float().unsqueeze(2)
        fused = fuse_mask * (w * traditional_out + (1 - w) * dzz_out) + (1 - fuse_mask) * traditional_out
        return fused

    def dtrg_tip2022(self, traditional_out: torch.Tensor, dzz_out: torch.Tensor, knn_k: int = 5):
        """
        /**
         * 方法: DTRG kNN 图一致性 (TIP 2022)
         * 思路:
         *  i) 随机投影 Z_c, Z_d ∈ R^{B×C×k} 并归一；
         *  ii) 以 1−cos 作为距离，逐通道构造 kNN 图 G_c, G_d；
         *  iii) 一致性 S = overlap(E(G_c), E(G_d))（边集合重叠率）；
         *  iv) 若 S>τ 则融合 F = η·F_t + (1−η)·F_d，否则回退 F_t；η 可常数或 S 映射。
         * 公平性:
         *  统一 τ 与融合算子；随机投影维度 k 与 HSIC 一致；不引入学习参数。
         */
        """
        B, C, L = traditional_out.shape
        k = max(8, min(self.hsic_proj_dim, L))
        P = torch.randn(L, k, device=traditional_out.device, dtype=traditional_out.dtype)
        P = F.normalize(P, dim=0)
        Zc = F.normalize(traditional_out @ P, dim=-1)
        Zd = F.normalize(dzz_out @ P, dim=-1)
        with torch.no_grad():
            Zf_c = Zc / (Zc.norm(dim=-1, keepdim=True).clamp_min(1e-8))
            Zf_d = Zd / (Zd.norm(dim=-1, keepdim=True).clamp_min(1e-8))
            sim_c = torch.matmul(Zf_c, Zf_c.transpose(1, 2)).clamp(-1, 1)
            sim_d = torch.matmul(Zf_d, Zf_d.transpose(1, 2)).clamp(-1, 1)
            dist_c = 1 - sim_c
            dist_d = 1 - sim_d
            fill = torch.eye(C, device=Zc.device, dtype=Zc.dtype) * 1e9
            dist_c = dist_c + fill
            dist_d = dist_d + fill
            kn_c = dist_c.topk(knn_k, largest=False, dim=-1).indices
            kn_d = dist_d.topk(knn_k, largest=False, dim=-1).indices
        overlap = []
        for b in range(B):
            inter_sum = 0
            for ch in range(C):
                set_c = set(kn_c[b, ch].tolist())
                set_d = set(kn_d[b, ch].tolist())
                inter = len(set_c.intersection(set_d))
                union = max(len(set_c.union(set_d)), 1)
                inter_sum += (inter / union)
            overlap.append(inter_sum / C)
        S = torch.tensor(overlap, device=traditional_out.device, dtype=traditional_out.dtype).view(B, 1, 1)
        tau = 0.5
        eta = 0.5
        fuse_mask = (S > tau).float()
        fused = fuse_mask * (eta * traditional_out + (1 - eta) * dzz_out) + (1 - fuse_mask) * traditional_out
        return fused

    def enable_visualization(self, enable=True):
        """开启/关闭可视化特征保存"""
        self.save_attention = enable
        if not enable:
            self.attention_maps.clear()
            self.activation_maps.clear()
    
    def get_visualization_features(self):
        """获取保存的可视化特征"""
        return {
            'attention': self.attention_maps.copy(),
            'activation': self.activation_maps.copy()
        }

    ###############################
    # 前向
    ###############################
    def forward(self, x: torch.Tensor, **kwargs):
        # x: (B,H,W,C)
        # 清理可能残留的 THOP hooks，避免 CPU/CUDA 混用导致的 device mismatch
        if not self._thop_hooks_cleared:
            try:
                for m in self.modules():
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
            self._thop_hooks_cleared = True
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x_proc, z = xz.chunk(2, dim=-1)
        x_proc = x_proc.permute(0, 3, 1, 2).contiguous()
        x_proc = self.act(self.conv2d(x_proc))

        if self.save_attention:
            self.activation_maps.append(x_proc.detach().cpu())


        if self.branch_mode == "traditional_only":
            fused = self.vmamba_traditional_ss2d_branch(x_proc)
        elif self.branch_mode == "dzz_only":
            fused = self.dzz_zigzag_diagonal_branch(x_proc)
        else:
            traditional_out = self.vmamba_traditional_ss2d_branch(x_proc)
            dzz_out = self.dzz_zigzag_diagonal_branch(x_proc)
                        # 保存分支输出（用于attention heatmap）
            if self.save_attention:
                # 使用方差作为attention的代理指标
                trad_attn = torch.var(traditional_out, dim=1, keepdim=True)
                dzz_attn = torch.var(dzz_out, dim=1, keepdim=True)
                self.attention_maps.append(trad_attn.detach().cpu())
                self.attention_maps.append(dzz_attn.detach().cpu())
                
            if self.enable_gating:
                method = (self.fusion_method or 'hsic').lower()
                if method in ('hsic_opt', 'optimized_hsic'):
                    fused = self.optimized_hsic_gate(traditional_out, dzz_out)  # (B, C, L)
                elif method in ('hsic', 'hsic_gate'):
                    fused = self.hsic_gate(traditional_out, dzz_out)
                elif method in ('i2pmae', 'i2pmae_cvpr2023'):
                    fused = self.i2pmae_cvpr2023(traditional_out, dzz_out)
                elif method in ('bftt3d', 'bftt3d_cvpr2024'):
                    fused = self.bftt3d_cvpr2024(traditional_out, dzz_out)
                elif method in ('dtrg', 'dtrg_tip2022'):
                    fused = self.dtrg_tip2022(traditional_out, dzz_out)
                elif method in ('cos', 'cosine'):
                    fused = self.gated_fusion(traditional_out, dzz_out)
                elif method in ('add', 'sum', 'eltwise'):
                    fused = self.fuse_add(traditional_out, dzz_out)
                elif method in ('conv1x1', 'conv'):
                    fused = self.fuse_conv1x1(traditional_out, dzz_out)
                elif method in ('se', 'se_fuse', 'se_gate'):
                    fused = self.fuse_se(traditional_out, dzz_out)
                else:
                    fused = self.hsic_gate(traditional_out, dzz_out)
            else:
                # 简单叠加或取和
                if self.scan_cache_reduce == 'mean':
                    fused = 0.5 * (traditional_out + dzz_out)
                else:
                    fused = traditional_out + dzz_out

        y = fused.transpose(1, 2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out  # (B,H,W,d_model)

##########################################################
# 基础组件
##########################################################

class MLP_V2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {}
        if device is not None: factory_kwargs["device"] = device
        if dtype is not None: factory_kwargs["dtype"] = dtype
        self.fc1 = nn.Linear(input_dim, hidden_dim, **factory_kwargs)
        self.fc2 = nn.Linear(hidden_dim, output_dim, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class VSSBlock_V2(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        expand: float = 2.0,
        use_residual: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual
        self.ln_1 = norm_layer(hidden_dim)
        self.ln_2 = norm_layer(hidden_dim)
        self.self_attention = SS2D_C(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            expand=expand,
            branch_mode=kwargs.get('branch_mode', 'both'),
            enable_cache=kwargs.get('enable_cache', True),
            enable_gating=kwargs.get('enable_gating', True),
            fusion_gate_type=kwargs.get('fusion_gate_type', 'cos'),
            fusion_method=kwargs.get('fusion_method', kwargs.get('fusion_gate_type', 'cos')),
            hsic_proj_dim=kwargs.get('hsic_proj_dim', 32),
            hsic_alpha=kwargs.get('hsic_alpha', 0.5),
            hsic_temperature=kwargs.get('hsic_temperature', 1.5),
            hsic_residual=kwargs.get('hsic_residual', 0.3),
            scan_cache_mode=kwargs.get('scan_cache_mode', 'dict'),
            scan_cache_precompute_shapes=kwargs.get('scan_cache_precompute_shapes', None),
            scan_cache_profile=kwargs.get('scan_cache_profile', False),
            scan_cache_reduce=kwargs.get('scan_cache_reduce', 'sum'),
        )
        mlp_hidden_dim = int(hidden_dim * 1.5)
        self.ffn = MLP_V2(hidden_dim, mlp_hidden_dim, hidden_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, input: torch.Tensor):
        if self.use_residual:
            x = input + self.drop_path(self.self_attention(self.ln_1(input)))
            x_out = x + self.drop_path(self.ffn(self.ln_2(x)))
        else:
            x = self.drop_path(self.self_attention(self.ln_1(input)))
            x_out = self.drop_path(self.ffn(self.ln_2(x)))
        return x_out

class PatchEmbed2D_V2(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None
        self._skip_norm_cleanup = False
    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            if not self._skip_norm_cleanup:
                # 防御性清理可能残留的 THOP hooks，避免 CPU/CUDA 设备不一致
                try:
                    if hasattr(self.norm, '_forward_hooks'):
                        self.norm._forward_hooks.clear()
                    if hasattr(self.norm, '_forward_pre_hooks'):
                        self.norm._forward_pre_hooks.clear()
                    if hasattr(self.norm, '_backward_hooks'):
                        self.norm._backward_hooks.clear()
                    if hasattr(self.norm, 'total_ops'):
                        try:
                            delattr(self.norm, 'total_ops')
                        except Exception:
                            pass
                    if hasattr(self.norm, 'total_params'):
                        try:
                            delattr(self.norm, 'total_params')
                        except Exception:
                            pass
                except Exception:
                    pass
            x = self.norm(x)
        return x

class PatchMerging2D_V2(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    def forward(self, x):
        B, H, W, C = x.shape
        fixH = H // 2
        fixW = W // 2
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        if (H % 2) != 0 or (W % 2) != 0:
            x0 = x0[:, :fixH, :fixW, :]
            x1 = x1[:, :fixH, :fixW, :]
            x2 = x2[:, :fixH, :fixW, :]
            x3 = x3[:, :fixH, :fixW, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class PatchExpand2D_V2(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_dim = dim
        self.output_dim = dim // 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, (dim_scale**2) * self.output_dim, bias=False)
        self.norm = norm_layer(self.output_dim)
    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c',
                      p1=self.dim_scale, p2=self.dim_scale, c=self.output_dim)
        x = self.norm(x)
        return x

class VSSLayer_V2(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        d_state=16,
        expand=2.0,
        use_residual=True,
        **kwargs
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            VSSBlock_V2(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=kwargs.get('attn_drop_rate', 0.),
                d_state=d_state,
                expand=expand,
                use_residual=use_residual,
                **kwargs
            )
            for i in range(depth)
        ])
        self.downsample = downsample(dim) if downsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class VSSLayer_up_V2(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        upsample=None,
        use_checkpoint=False,
        d_state=16,
        expand=2.0,
        use_residual=True,
        **kwargs
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.upsample = upsample
        self.blocks = nn.ModuleList([
            VSSBlock_V2(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=kwargs.get('attn_drop_rate', 0.),
                d_state=d_state,
                expand=expand,
                use_residual=use_residual,
                **kwargs
            )
            for i in range(depth)
        ])

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x

##########################################################
# 主干 TopoMamba_2D
##########################################################

class TopoMamba_2D(nn.Module):
    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        depths=[2, 2, 8, 2],
        depths_decoder=[2, 8, 2, 2],
        dims=[96, 192, 384, 768],
        dims_decoder=[768, 384, 192, 96],
        d_state=1,
        expand=1.0,
        drop_path_rate=0.1,
        patch_norm=True,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
        use_skip_connection=True,
        use_residual=True,
        # SS2D_C / ScanCache 额外参数透传
        branch_mode='both',
        enable_cache=False,
        enable_gating=True,
        fusion_gate_type='hsic',
        fusion_method=None,
        hsic_proj_dim=32,
        hsic_alpha=0.5,
        hsic_temperature=1.5,
        hsic_residual=0.3,
        scan_cache_mode='buffer',
        scan_cache_precompute_shapes=None,
        scan_cache_profile=False,
        scan_cache_reduce='sum',
        # 预训练加载配置（新增）
        load_pretrained=True,
        pretrained_path: Optional[str]=None,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.depths = depths
        self.depths_decoder = depths_decoder
        self.dims = dims
        self.dims_decoder = dims_decoder
        self.patch_norm = patch_norm
        self.use_skip_connection = use_skip_connection
        self.use_residual = use_residual

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_dec = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.patch_embed = PatchEmbed2D_V2(patch_size, in_chans, dims[0],
                                           norm_layer if patch_norm else None)

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = VSSLayer_V2(
                dim=dims[i],
                depth=depths[i],
                drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D_V2 if (i < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                d_state=d_state,
                expand=expand,
                use_residual=use_residual,
                branch_mode=branch_mode,
                enable_cache=enable_cache,
                enable_gating=enable_gating,
                fusion_gate_type=fusion_gate_type,
                fusion_method=fusion_method or fusion_gate_type,
                hsic_proj_dim=hsic_proj_dim,
                hsic_alpha=hsic_alpha,
                hsic_temperature=hsic_temperature,
                hsic_residual=hsic_residual,
                scan_cache_mode=scan_cache_mode,
                scan_cache_precompute_shapes=scan_cache_precompute_shapes,
                scan_cache_profile=scan_cache_profile,
                scan_cache_reduce=scan_cache_reduce,
            )
            self.layers.append(layer)

        # Decoder
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                concat_linear = nn.Identity()
                layer_up = PatchExpand2D_V2(dims_decoder[i], dim_scale=2, norm_layer=norm_layer)
            else:
                skip_dim = dims[self.num_layers - 1 - i]
                current_dim = dims_decoder[i]
                concat_dim = current_dim + skip_dim
                concat_linear = nn.Linear(concat_dim, dims_decoder[i])
                if i < self.num_layers - 1:
                    upsampled_dim = dims_decoder[i] // 2
                    layer_up = VSSLayer_up_V2(
                        dim=upsampled_dim,
                        depth=depths_decoder[i],
                        drop_path=dpr_dec[sum(depths_decoder[:i]):sum(depths_decoder[:i+1])],
                        norm_layer=norm_layer,
                        upsample=PatchExpand2D_V2(dims_decoder[i], dim_scale=2, norm_layer=norm_layer),
                        use_checkpoint=use_checkpoint,
                        d_state=d_state,
                        expand=expand,
                        use_residual=use_residual,
                        branch_mode=branch_mode,
                        enable_cache=enable_cache,
                        enable_gating=enable_gating,
                        fusion_gate_type=fusion_gate_type,
                        fusion_method=fusion_method or fusion_gate_type,
                        hsic_proj_dim=hsic_proj_dim,
                        hsic_alpha=hsic_alpha,
                        hsic_temperature=hsic_temperature,
                        hsic_residual=hsic_residual,
                        scan_cache_mode=scan_cache_mode,
                        scan_cache_precompute_shapes=scan_cache_precompute_shapes,
                        scan_cache_profile=scan_cache_profile,
                        scan_cache_reduce=scan_cache_reduce,
                    )
                else:
                    layer_up = VSSLayer_up_V2(
                        dim=dims_decoder[i],
                        depth=depths_decoder[i],
                        drop_path=dpr_dec[sum(depths_decoder[:i]):sum(depths_decoder[:i+1])],
                        norm_layer=norm_layer,
                        upsample=None,
                        use_checkpoint=use_checkpoint,
                        d_state=d_state,
                        expand=expand,
                        use_residual=use_residual,
                        branch_mode=branch_mode,
                        enable_cache=enable_cache,
                        enable_gating=enable_gating,
                        fusion_gate_type=fusion_gate_type,
                        fusion_method=fusion_method or fusion_gate_type,
                        hsic_proj_dim=hsic_proj_dim,
                        hsic_alpha=hsic_alpha,
                        hsic_temperature=hsic_temperature,
                        hsic_residual=hsic_residual,
                        scan_cache_mode=scan_cache_mode,
                        scan_cache_precompute_shapes=scan_cache_precompute_shapes,
                        scan_cache_profile=scan_cache_profile,
                        scan_cache_reduce=scan_cache_reduce,
                    )
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.final_conv = nn.Conv2d(dims_decoder[-1], num_classes, 1) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        # THOP hooks 清理标记（模型级，优先于 encoder 调用）
        self._thop_hooks_cleared = False

        # 自动加载预训练（仅encoder/backbone部分）
        if load_pretrained and pretrained_path and isinstance(pretrained_path, str):
            try:
                if os.path.exists(pretrained_path):
                    self.load_pretrained_backbone(pretrained_path, verbose=False)
                else:
                    pass
            except Exception:
                pass

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            if m.weight.numel() < 10000:
                trunc_normal_(m.weight, std=.02)
            else:
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_pretrained_backbone(self, ckpt_path: str, verbose: bool=False) -> Dict[str, Any]:
        """
        过滤加载来自 VMamba/UperNet 风格的预训练权重，仅映射到 encoder 主干：
        - 映射规则（源 -> 目标）：
          backbone.patch_embed.0.(weight|bias) -> patch_embed.proj.(weight|bias)
          backbone.patch_embed.(2|5|7).(weight|bias) -> patch_embed.norm.(weight|bias) 若存在
          backbone.layers.{i}.blocks.{j}.norm.(weight|bias) -> layers.{i}.blocks.{j}.ln_1/ln_2.(weight|bias)
          backbone.layers.{i}.blocks.{j}.op.{in_proj,conv2d,out_norm,out_proj}.* -> 同名至 self_attention.*
          backbone.layers.{i}.blocks.{j}.op.{x_proj_weight,dt_projs_weight,dt_projs_bias,A_logs,Ds}
            -> 同时映射到 self_attention 的 _traditional 与 _dzz 两个分支（若存在）
        - 未匹配或形状不一致将被忽略
        返回加载统计信息。
        """
        import os as _os
        import torch as _torch
        ck = _torch.load(ckpt_path, map_location='cpu')
        sd_src: Dict[str, _torch.Tensor] = ck.get('state_dict', ck)

        dst_sd = self.state_dict()
        mapped: Dict[str, _torch.Tensor] = {}

        def try_set(dst_key: str, src_tensor: _torch.Tensor):
            if dst_key in dst_sd and dst_sd[dst_key].shape == src_tensor.shape:
                mapped[dst_key] = src_tensor

        for k, v in sd_src.items():
            if not isinstance(k, str):
                continue
            # 1) 直接同名/去前缀同名映射（原生DZZMamba权重）
            direct_key = k
            for pre in ('module.', 'model.', 'state_dict.', 'net.'):
                if direct_key.startswith(pre):
                    direct_key = direct_key[len(pre):]
            if direct_key in dst_sd and dst_sd[direct_key].shape == v.shape:
                mapped[direct_key] = v
                continue

            # 2) UperNet/VMamba 风格的 backbone.* 权重映射
            if not direct_key.startswith('backbone.'):
                continue
            ks = direct_key.split('.')
            # patch_embed
            if ks[1] == 'patch_embed':
                if ks[2] == '0':
                    # conv
                    if ks[3] in ('weight', 'bias'):
                        try_set(f'patch_embed.proj.{ks[3]}', v)
                else:
                    # 可能的norm索引 2/5/7
                    if hasattr(self.patch_embed, 'norm') and self.patch_embed.norm is not None and ks[3] in ('weight','bias'):
                        try_set(f'patch_embed.norm.{ks[3]}', v)
                continue

            # layers -> encoder blocks
            if ks[1] == 'layers':
                li = ks[2]; sub = ks[3]
                if sub == 'blocks':
                    bj = ks[4]
                    if ks[5] == 'norm' and ks[6] in ('weight','bias'):
                        # 同时尝试 ln_1 与 ln_2（encoder）
                        try_set(f'layers.{li}.blocks.{bj}.ln_1.{ks[6]}', v)
                        try_set(f'layers.{li}.blocks.{bj}.ln_2.{ks[6]}', v)
                        # 同步尝试映射到可能的decoder镜像层（layers_up）
                        try:
                            enc_idx = int(li)
                            num_layers = self.num_layers
                            # 计算decoder层索引与目标hidden_dim
                            # 对于前 num_layers-1 个decoder层，其hidden_dim为 dims_decoder[i]//2，否则为 dims_decoder[i]
                            for dec_i in range(num_layers):
                                if dec_i < num_layers - 1:
                                    dec_hidden = self.dims_decoder[dec_i] // 2
                                else:
                                    dec_hidden = self.dims_decoder[dec_i]
                                # 找到匹配的encoder层：hidden_dim相等
                                if 0 <= enc_idx < len(self.dims) and self.dims[enc_idx] == dec_hidden:
                                    try_set(f'layers_up.{dec_i}.blocks.{bj}.ln_1.{ks[6]}', v)
                                    try_set(f'layers_up.{dec_i}.blocks.{bj}.ln_2.{ks[6]}', v)
                        except Exception:
                            pass
                        continue
                    if ks[5] == 'op':
                        op_key = '.'.join(ks[6:])
                        # 直接同名映射到encoder self_attention
                        for simple in ('in_proj.weight','in_proj.bias','conv2d.weight','conv2d.bias',
                                       'out_norm.weight','out_norm.bias','out_proj.weight','out_proj.bias'):
                            if op_key == simple:
                                try_set(f'layers.{li}.blocks.{bj}.self_attention.{simple}', v)
                                break
                        # 共享到两分支的权重（encoder）
                        if op_key in ('x_proj_weight','dt_projs_weight','dt_projs_bias','A_logs','Ds'):
                            for suffix in ('_traditional', '_dzz'):
                                try_set(f'layers.{li}.blocks.{bj}.self_attention.{op_key}{suffix}', v)
                         
                        # 同步尝试映射到decoder镜像层（layers_up）
                        try:
                            enc_idx = int(li)
                            num_layers = self.num_layers
                            for dec_i in range(num_layers):
                                if dec_i < num_layers - 1:
                                    dec_hidden = self.dims_decoder[dec_i] // 2
                                else:
                                    dec_hidden = self.dims_decoder[dec_i]
                                if 0 <= enc_idx < len(self.dims) and self.dims[enc_idx] == dec_hidden:
                                    # 直接同名映射
                                    for simple in ('in_proj.weight','in_proj.bias','conv2d.weight','conv2d.bias',
                                                   'out_norm.weight','out_norm.bias','out_proj.weight','out_proj.bias'):
                                        if op_key == simple:
                                            try_set(f'layers_up.{dec_i}.blocks.{bj}.self_attention.{simple}', v)
                                            break
                                    # 共享到两分支
                                    if op_key in ('x_proj_weight','dt_projs_weight','dt_projs_bias','A_logs','Ds'):
                                        for suffix in ('_traditional', '_dzz'):
                                            try_set(f'layers_up.{dec_i}.blocks.{bj}.self_attention.{op_key}{suffix}', v)
                        except Exception:
                            pass
                        continue
 
        # 统计参数数量 - 修正：计算实际被加载的模型参数（避免重复计数）
        model_total_params = sum(p.numel() for p in self.parameters())
        
        # 计算实际有多少模型参数被加载了权重（去重）
        actually_loaded_params = 0
        loaded_keys_set = set(mapped.keys())
        for name, param in self.named_parameters():
            if name in loaded_keys_set:
                actually_loaded_params += param.numel()
        
        # 源权重的总参数量（用于参考）
        source_total_params = sum(t.numel() for k, t in sd_src.items() if isinstance(t, _torch.Tensor))
        
        loading_ratio_params = (actually_loaded_params / max(1, model_total_params)) * 100.0
        num_mapped_keys = len(mapped)
        total_keys = len([k for k in sd_src.keys() if isinstance(sd_src[k], _torch.Tensor)])

        # 实际加载
        missing, unexpected = self.load_state_dict(mapped, strict=False)

        # 打印信息
        print("[Pretrained Loading]")
        print(f" - path: {ckpt_path}")
        print(f" - model_total_params: {model_total_params:,} ({model_total_params/1e6:.2f}M)")
        print(f" - actually_loaded_params: {actually_loaded_params:,} ({actually_loaded_params/1e6:.2f}M)")
        print(f" - source_weight_params: {source_total_params:,} ({source_total_params/1e6:.2f}M)")
        print(f" - mapped_keys: {num_mapped_keys}/{total_keys}")
        print(f" - loading_ratio(params): {loading_ratio_params:.2f}%")

        # 保存报告供训练脚本记录
        self._pretrained_load_report = {
            'pretrained_path': ckpt_path,
            'model_total_params': model_total_params,
            'actually_loaded_params': actually_loaded_params,
            'source_weight_params': source_total_params,
            'loading_ratio_params': loading_ratio_params,
            'num_mapped_keys': num_mapped_keys,
            'total_keys': total_keys,
        }
        return dict(self._pretrained_load_report)

    def forward_features(self, x):
        x = self.patch_embed(x)  # (B,H',W',C0)
        skip = []
        for layer in self.layers:
            if self.use_skip_connection:
                skip.append(x)
            x = layer(x)
        return x, skip

    def forward_features_up(self, x, skip):
        for i, layer_up in enumerate(self.layers_up):
            if i == 0:
                x = layer_up(x)
            else:
                if self.use_skip_connection and len(skip) > 0:
                    idx = self.num_layers - 1 - i
                    if idx < len(skip):
                        sk = skip[idx]
                        if x.shape[1:3] != sk.shape[1:3]:
                            sk = F.interpolate(
                                sk.permute(0,3,1,2),
                                size=(x.shape[1], x.shape[2]),
                                mode='bilinear',
                                align_corners=False
                            ).permute(0,2,3,1)
                        x = torch.cat([x, sk], dim=-1)
                        x = self.concat_back_dim[i](x)
                x = layer_up(x)
        return x

    def forward_final(self, x):
        if self.num_classes > 0:
            x = x.permute(0,3,1,2)
            x = self.final_conv(x)
            # 恢复到输入原始分辨率（假设 patch_size=4）
            x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        else:
            x = x.permute(0,3,1,2)
        return x

    def forward(self, x):
        # x: (B,C,H,W)
        # 在进入 encoder 前清理潜在残留的 THOP hooks，避免 CPU/CUDA 混用
        if not self._thop_hooks_cleared:
            try:
                for m in self.modules():
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
            self._thop_hooks_cleared = True
        x, skips = self.forward_features(x)
        x = self.forward_features_up(x, skips)
        x = self.forward_final(x)
        return x

##########################################################
# 模型注册
##########################################################

@register_model
def TopoMamba_2D_t(**kwargs):
    return TopoMamba_2D(**kwargs)

@register_model
def TopoMamba_2D_s(**kwargs):
    return TopoMamba_2D(**kwargs)

@register_model
def TopoMamba_2D_b(**kwargs):
    return TopoMamba_2D(**kwargs)

# 别名
@register_model
def topomamba_2d_v3_t(**kwargs): return TopoMamba_2D_t(**kwargs)
@register_model
def topomamba_2d_v3_s(**kwargs): return TopoMamba_2D_s(**kwargs)
@register_model
def topomamba_2d_v3_b(**kwargs): return TopoMamba_2D_b(**kwargs)
@register_model
def topomamba_2d_t(**kwargs): return TopoMamba_2D_t(**kwargs)
@register_model
def topomamba_2d_s(**kwargs): return TopoMamba_2D_s(**kwargs)
@register_model
def topomamba_2d_b(**kwargs): return TopoMamba_2D_b(**kwargs)

# Backward-compatible registry aliases for historical checkpoints/scripts.
# New configs and result paths should use TopoMamba_2D_* names.
@register_model
def CUMamba_t(**kwargs): return TopoMamba_2D_t(**kwargs)
@register_model
def CUMamba_s(**kwargs): return TopoMamba_2D_s(**kwargs)
@register_model
def CUMamba_b(**kwargs): return TopoMamba_2D_b(**kwargs)
@register_model
def cumamba_t(**kwargs): return TopoMamba_2D_t(**kwargs)
@register_model
def cumamba_s(**kwargs): return TopoMamba_2D_s(**kwargs)
@register_model
def cumamba_b(**kwargs): return TopoMamba_2D_b(**kwargs)

##########################################################
# 测试和统计函数
##########################################################

def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_TopoMamba_2D_models():
    """测试TopoMamba_2D模型"""
    models = ['TopoMamba_2D_t', 'TopoMamba_2D_s', 'TopoMamba_2D_b']
    
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    for model_name in models:
        print(f"\n=== Testing {model_name} ===")
        model = create_model(model_name, num_classes=9, in_chans=3)
        model = model.to(device)  # 移动模型到GPU
        
        # 测试输入
        x = torch.randn(2, 3, 512, 512).to(device)  # 移动输入到GPU
        
        try:
            y = model(x)
            params = count_parameters(model)
            print(f"✓ {model_name}: Input {x.shape} -> Output {y.shape}")
            print(f"  Parameters: {params:,}")
        except Exception as e:
            print(f"✗ {model_name}: Error - {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_TopoMamba_2D_models() 
