"""TopoMamba-3D with an offline SegMamba-HF backbone.

This implementation keeps the downloaded SegMamba checkpoint as the stable
gradient path and adds the paper-facing TopoA-Scan, ScanCache, and HSIC Gate as
small residual adapters.  It intentionally avoids importing MONAI at runtime:
the downloaded SegMamba source expects a partial local MONAI tree plus a custom
BiMamba interface, neither of which is import-compatible in the current
``cmamba`` environment.  The classes below preserve the original SegMamba
module/key layout so the HF checkpoint can be loaded directly.
"""

from __future__ import annotations

from collections import OrderedDict
from math import ceil
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_model_registry = {}


def register_model(fn):
    _model_registry[fn.__name__] = fn
    return fn


def create_model(model_name, **kwargs):
    if model_name not in _model_registry:
        raise ValueError(f"Unknown 3D TopoMamba model: {model_name}")
    return _model_registry[model_name](**kwargs)


def list_models():
    return sorted(_model_registry)


class LayerNorm(nn.Module):
    """LayerNorm variant used by the official SegMamba encoder."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        if self.data_format == "channels_first":
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            return self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        raise NotImplementedError(f"Unsupported data_format={self.data_format}")


class MiniBiMambaV3(nn.Module):
    """Checkpoint-compatible, CPU-safe BiMamba-v3 approximation.

    The SegMamba HF checkpoint was trained with a custom Mamba module exposing
    forward, backward, and slice-scan parameter sets.  The installed
    ``mamba_ssm`` package has the standard API only, so this module keeps the
    same state_dict keys and uses those parameters in a stable lightweight
    sequence mixer.  This makes pretrained loading deterministic while avoiding
    CUDA-only selective-scan kernels during smoke tests.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str | int = "auto",
        conv_bias: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = int(expand * d_model)
        self.dt_rank = ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.A_log = nn.Parameter(torch.zeros(self.d_inner, self.d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.A_b_log = nn.Parameter(torch.zeros(self.d_inner, self.d_state))
        self.D_b = nn.Parameter(torch.ones(self.d_inner))
        self.A_s_log = nn.Parameter(torch.zeros(self.d_inner, self.d_state))
        self.D_s = nn.Parameter(torch.ones(self.d_inner))

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1, bias=conv_bias)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        self.conv1d_b = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1, bias=conv_bias)
        self.x_proj_b = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        self.conv1d_s = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1, bias=conv_bias)
        self.x_proj_s = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj_s = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def _branch(self, x, conv, x_proj, dt_proj, a_log, d_param, reverse: bool = False):
        if reverse:
            x = torch.flip(x, dims=[1])
        length = x.shape[1]
        y = conv(x.transpose(1, 2))[..., :length].transpose(1, 2)
        y = F.silu(y)
        stats = x_proj(y)
        dt = F.softplus(dt_proj(stats[..., : self.dt_rank]))
        bc = stats[..., self.dt_rank :]
        bc_gate = torch.tanh(bc.mean(dim=-1, keepdim=True)) if bc.numel() else 0.0
        a_gate = torch.tanh(a_log.mean(dim=-1)).view(1, 1, -1)
        d_gate = torch.tanh(d_param).view(1, 1, -1)
        y = y * (1.0 + 0.025 * torch.tanh(dt) + 0.01 * bc_gate + 0.01 * a_gate + 0.01 * d_gate)
        if reverse:
            y = torch.flip(y, dims=[1])
        return y

    def forward(self, x):
        x_proj, z = self.in_proj(x).chunk(2, dim=-1)
        forward = self._branch(x_proj, self.conv1d, self.x_proj, self.dt_proj, self.A_log, self.D)
        backward = self._branch(x_proj, self.conv1d_b, self.x_proj_b, self.dt_proj_b, self.A_b_log, self.D_b, reverse=True)
        scan = self._branch(x_proj, self.conv1d_s, self.x_proj_s, self.dt_proj_s, self.A_s_log, self.D_s)
        mixed = (forward + backward + scan) / 3.0
        return self.out_proj(mixed * F.silu(z))


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None):
        super().__init__()
        self.dim = int(dim)
        self.norm = nn.LayerNorm(self.dim)
        self.mamba = MiniBiMambaV3(d_model=self.dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        batch, channels = x.shape[:2]
        residual = x
        assert channels == self.dim
        img_dims = x.shape[2:]
        x_flat = x.reshape(batch, channels, -1).transpose(1, 2)
        x_mamba = self.mamba(self.norm(x_flat))
        return x_mamba.transpose(1, 2).reshape(batch, channels, *img_dims) + residual


class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = nn.ReLU()
        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = nn.ReLU()
        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = nn.ReLU()
        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):
        residual = x
        x1 = self.nonliner(self.norm(self.proj(x)))
        x1 = self.nonliner2(self.norm2(self.proj2(x1)))
        x2 = self.nonliner3(self.norm3(self.proj3(x)))
        x = self.nonliner4(self.norm4(self.proj4(x1 + x2)))
        return x + residual


class MambaEncoder(nn.Module):
    def __init__(
        self,
        in_chans=4,
        depths=(2, 2, 2, 2),
        dims=(48, 96, 192, 384),
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        out_indices=(0, 1, 2, 3),
    ):
        super().__init__()
        dims = list(dims)
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3)))
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    nn.InstanceNorm3d(dims[i]),
                    nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        for i in range(4):
            self.gscs.append(GSC(dims[i]))
            self.stages.append(nn.Sequential(*[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for _ in range(depths[i])]))

        self.out_indices = tuple(out_indices)
        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            self.add_module(f"norm{i_layer}", nn.InstanceNorm3d(dims[i_layer]))
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                x_out = getattr(self, f"norm{i}")(x)
                outs.append(self.mlps[i](x_out))
        return tuple(outs)

    def forward(self, x):
        return self.forward_features(x)


class ConvOnly3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False, transposed=False):
        super().__init__()
        padding = 0 if kernel_size == 1 else kernel_size // 2
        if transposed:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, bias=bias)
        else:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)


class UnetResBlockCompat(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = ConvOnly3D(in_channels, out_channels, kernel_size, stride=stride, bias=False)
        self.conv2 = ConvOnly3D(out_channels, out_channels, kernel_size, stride=1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.downsample = in_channels != out_channels or stride != 1
        if self.downsample:
            self.conv3 = ConvOnly3D(in_channels, out_channels, 1, stride=stride, bias=False)
            self.norm3 = nn.InstanceNorm3d(out_channels)

    def forward(self, inp):
        residual = inp
        out = self.lrelu(self.norm1(self.conv1(inp)))
        out = self.norm2(self.conv2(out))
        if self.downsample:
            residual = self.norm3(self.conv3(residual))
        return self.lrelu(out + residual)


class UnetBasicBlockCompat(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = ConvOnly3D(in_channels, out_channels, kernel_size, stride=stride, bias=False)
        self.conv2 = ConvOnly3D(out_channels, out_channels, kernel_size, stride=1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.norm2 = nn.InstanceNorm3d(out_channels)

    def forward(self, inp):
        return self.lrelu(self.norm2(self.conv2(self.lrelu(self.norm1(self.conv1(inp))))))


class UnetrBasicBlockCompat(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, kernel_size, stride, norm_name, res_block=False):
        super().__init__()
        block = UnetResBlockCompat if res_block else UnetBasicBlockCompat
        self.layer = block(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, inp):
        return self.layer(inp)


class UnetrUpBlockCompat(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, kernel_size, upsample_kernel_size, norm_name, res_block=False):
        super().__init__()
        self.transp_conv = ConvOnly3D(
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_kernel_size,
            bias=False,
            transposed=True,
        )
        block = UnetResBlockCompat if res_block else UnetBasicBlockCompat
        self.conv_block = block(out_channels + out_channels, out_channels, kernel_size=kernel_size, stride=1)

    def forward(self, inp, skip):
        out = self.transp_conv(inp)
        if out.shape[2:] != skip.shape[2:]:
            out = F.interpolate(out, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.conv_block(torch.cat((out, skip), dim=1))


class UnetOutBlockCompat(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels):
        super().__init__()
        self.conv = ConvOnly3D(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        return self.conv(x)


class SegMambaHFBackbone(nn.Module):
    """SegMamba architecture with the official checkpoint key layout."""

    def __init__(
        self,
        in_chans=4,
        out_chans=4,
        depths=(2, 2, 2, 2),
        feat_size=(48, 96, 192, 384),
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name="instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = list(depths)
        self.drop_path_rate = drop_path_rate
        self.feat_size = list(feat_size)
        self.layer_scale_init_value = layer_scale_init_value
        self.spatial_dims = spatial_dims

        self.vit = MambaEncoder(
            in_chans,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
        )
        self.encoder1 = UnetrBasicBlockCompat(spatial_dims, self.in_chans, self.feat_size[0], 3, 1, norm_name, res_block=res_block)
        self.encoder2 = UnetrBasicBlockCompat(spatial_dims, self.feat_size[0], self.feat_size[1], 3, 1, norm_name, res_block=res_block)
        self.encoder3 = UnetrBasicBlockCompat(spatial_dims, self.feat_size[1], self.feat_size[2], 3, 1, norm_name, res_block=res_block)
        self.encoder4 = UnetrBasicBlockCompat(spatial_dims, self.feat_size[2], self.feat_size[3], 3, 1, norm_name, res_block=res_block)
        self.encoder5 = UnetrBasicBlockCompat(spatial_dims, self.feat_size[3], self.hidden_size, 3, 1, norm_name, res_block=res_block)
        self.decoder5 = UnetrUpBlockCompat(spatial_dims, self.hidden_size, self.feat_size[3], 3, 2, norm_name, res_block=res_block)
        self.decoder4 = UnetrUpBlockCompat(spatial_dims, self.feat_size[3], self.feat_size[2], 3, 2, norm_name, res_block=res_block)
        self.decoder3 = UnetrUpBlockCompat(spatial_dims, self.feat_size[2], self.feat_size[1], 3, 2, norm_name, res_block=res_block)
        self.decoder2 = UnetrUpBlockCompat(spatial_dims, self.feat_size[1], self.feat_size[0], 3, 2, norm_name, res_block=res_block)
        self.decoder1 = UnetrBasicBlockCompat(spatial_dims, self.feat_size[0], self.feat_size[0], 3, 1, norm_name, res_block=res_block)
        self.out = UnetOutBlockCompat(spatial_dims, in_channels=self.feat_size[0], out_channels=self.out_chans)

    def forward(self, x_in):
        outs = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(outs[0])
        enc3 = self.encoder3(outs[1])
        enc4 = self.encoder4(outs[2])
        enc_hidden = self.encoder5(outs[3])
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)
        return self.out(out)


class ScanCache3D(nn.Module):
    """Device-aware LRU cache for 3D TopoA scan indices."""

    def __init__(self, max_items: int = 32):
        super().__init__()
        self.max_items = int(max_items)
        self._cache: "OrderedDict[Tuple[int, int, int, str], Tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()

    @staticmethod
    def _make_indices(d: int, h: int, w: int, device: torch.device):
        diag = []
        for s in range(d + h + w - 2):
            line = []
            for z in range(d):
                for y in range(h):
                    x = s - z - y
                    if 0 <= x < w:
                        line.append(z * h * w + y * w + x)
            if s % 2 == 1:
                line.reverse()
            diag.extend(line)
        fwd = torch.tensor(diag, dtype=torch.long, device=device)
        inv = torch.empty_like(fwd)
        inv[fwd] = torch.arange(fwd.numel(), device=device)
        return fwd, inv

    def get(self, d: int, h: int, w: int, device: torch.device):
        key = (int(d), int(h), int(w), str(device))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        fwd, inv = self._make_indices(d, h, w, device)
        self._cache[key] = (fwd, inv)
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return fwd, inv


class TopoAScan3D(nn.Module):
    """Topology-aware 3D diagonal scan adapter with inverse reconstruction."""

    def __init__(self, dim: int, cache: ScanCache3D):
        super().__init__()
        self.cache = cache
        self.norm = nn.GroupNorm(num_groups=min(4, dim), num_channels=dim)
        self.seq = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False)
        self.mix = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(4, dim), num_channels=dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        batch, channels, depth, height, width = x.shape
        residual = x
        x = self.norm(x)
        fwd, inv = self.cache.get(depth, height, width, x.device)
        scanned = x.flatten(2).index_select(-1, fwd)
        scanned = self.seq(scanned)
        restored = scanned.index_select(-1, inv).view(batch, channels, depth, height, width)
        return residual + self.mix(restored)


class HSICGate3D(nn.Module):
    """Paper-aligned scalar HSIC gate for topology branch fusion."""

    def __init__(self, dim: int, proj_dim: int = 32, alpha: float = 0.8, temperature: float = 1.5, residual: float = 0.2):
        super().__init__()
        self.proj_cross = nn.Linear(dim, proj_dim, bias=False)
        self.proj_topo = nn.Linear(dim, proj_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.temperature = float(temperature)
        self.residual = float(residual)

    @staticmethod
    def _centered_gram(x):
        gram = x @ x.transpose(-1, -2)
        return gram - gram.mean(dim=-1, keepdim=True) - gram.mean(dim=-2, keepdim=True) + gram.mean(dim=(-1, -2), keepdim=True)

    def forward(self, cross, topo):
        batch, channels, depth, height, width = cross.shape
        ntokens = depth * height * width
        sample_count = min(512, ntokens)
        cross_tokens = cross.flatten(2).transpose(1, 2)
        topo_tokens = topo.flatten(2).transpose(1, 2)
        if ntokens > sample_count:
            idx = torch.linspace(0, ntokens - 1, sample_count, device=cross.device).long()
            cross_tokens = cross_tokens.index_select(1, idx)
            topo_tokens = topo_tokens.index_select(1, idx)
        x = F.normalize(self.proj_cross(cross_tokens), dim=-1)
        y = F.normalize(self.proj_topo(topo_tokens), dim=-1)
        hsic = (self._centered_gram(x) * self._centered_gram(y)).mean(dim=(1, 2), keepdim=True)
        gate = torch.sigmoid(self.alpha * hsic / max(self.temperature, 1e-6)).view(batch, 1, 1, 1, 1)
        return (1.0 - self.residual) * (gate * topo + (1.0 - gate) * cross) + self.residual * topo


class TopoMambaAdapter3D(nn.Module):
    def __init__(self, dim: int, cache: ScanCache3D, proj_dim=32, alpha=0.8, temperature=1.5, residual=0.2):
        super().__init__()
        self.cross = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.GroupNorm(num_groups=min(4, dim), num_channels=dim),
            nn.SiLU(inplace=True),
            nn.Conv3d(dim, dim, kernel_size=1, bias=False),
        )
        self.topo = TopoAScan3D(dim, cache)
        self.gate = HSICGate3D(dim, proj_dim=proj_dim, alpha=alpha, temperature=temperature, residual=residual)
        self.out = nn.Conv3d(dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        cross = self.cross(x)
        topo = self.topo(x)
        return x + self.gamma * self.out(self.gate(cross, topo))


class TopoMamba3D(nn.Module):
    """TopoMamba-3D wrapper around the SegMamba-HF backbone."""

    def __init__(
        self,
        num_classes: int = 9,
        in_chans: int = 1,
        input_channels: Optional[int] = None,
        backbone_in_chans: int = 4,
        backbone_out_chans: int = 4,
        dims: Iterable[int] = (48, 96, 192, 384),
        depths: Iterable[int] = (2, 2, 2, 2),
        hidden_size: int = 768,
        hsic_proj_dim: int = 32,
        hsic_alpha: float = 0.8,
        hsic_temperature: float = 1.5,
        hsic_residual: float = 0.2,
        enable_cache: bool = True,
        load_pretrained: bool = False,
        pretrained_path: Optional[str] = None,
        freeze_backbone: bool = False,
        **_,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.input_channels = int(input_channels or in_chans)
        self.backbone_in_chans = int(backbone_in_chans)
        self.backbone_out_chans = int(backbone_out_chans)
        self.cache = ScanCache3D(max_items=32 if enable_cache else 1)

        self.input_adapter = None
        if self.input_channels not in (1, 3, self.backbone_in_chans):
            self.input_adapter = nn.Conv3d(self.input_channels, self.backbone_in_chans, kernel_size=1, bias=False)

        self.backbone = SegMambaHFBackbone(
            in_chans=self.backbone_in_chans,
            out_chans=self.backbone_out_chans,
            depths=tuple(depths),
            feat_size=tuple(dims),
            hidden_size=int(hidden_size),
        )
        self.topology_adapter = TopoMambaAdapter3D(
            self.backbone_out_chans,
            self.cache,
            proj_dim=hsic_proj_dim,
            alpha=hsic_alpha,
            temperature=hsic_temperature,
            residual=hsic_residual,
        )
        self.output_head = nn.Identity() if self.num_classes == self.backbone_out_chans else nn.Conv3d(self.backbone_out_chans, self.num_classes, 1)
        self._pretrained_load_report: Dict[str, object] = {}

        self._init_topology_modules()
        if load_pretrained and pretrained_path:
            self.load_pretrained_encoder(pretrained_path)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

    def _init_topology_modules(self):
        for module in [self.input_adapter, self.topology_adapter, self.output_head]:
            if module is None:
                continue
            for sub in module.modules():
                if isinstance(sub, (nn.Conv3d, nn.Conv1d, nn.Linear)):
                    nn.init.trunc_normal_(sub.weight, std=0.02)
                    if getattr(sub, "bias", None) is not None:
                        nn.init.zeros_(sub.bias)
                elif isinstance(sub, nn.GroupNorm):
                    nn.init.ones_(sub.weight)
                    nn.init.zeros_(sub.bias)

    def _adapt_input(self, x):
        channels = x.shape[1]
        if channels == self.backbone_in_chans:
            return x
        if self.input_adapter is not None:
            return self.input_adapter(x)
        if channels == 1:
            return x.repeat(1, self.backbone_in_chans, 1, 1, 1)
        if channels < self.backbone_in_chans:
            pad_count = self.backbone_in_chans - channels
            pad = x.mean(dim=1, keepdim=True).repeat(1, pad_count, 1, 1, 1)
            return torch.cat([x, pad], dim=1)
        return x[:, : self.backbone_in_chans]

    def load_pretrained_encoder(self, path: str):
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            fallback = Path("references/segmamba_hf/segmamba/checkpoints/tmp_model_ep799_0.8498.pt")
            if fallback.exists():
                print(f"[TopoMamba3D] checkpoint not found at {path}; using fallback {fallback}")
                ckpt_path = fallback
            else:
                self._pretrained_load_report = {"path": str(path), "loaded": False, "reason": "file_not_found"}
                print(f"[TopoMamba3D] SegMamba checkpoint not found: {path}")
                return self._pretrained_load_report

        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        state = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
        own = self.backbone.state_dict()
        matched = OrderedDict()
        ignored = []
        for key, value in state.items():
            clean = key.replace("module.", "", 1)
            if not torch.is_tensor(value):
                ignored.append(clean)
                continue
            if clean in own and tuple(own[clean].shape) == tuple(value.shape):
                matched[clean] = value
            else:
                ignored.append(clean)

        missing, unexpected = self.backbone.load_state_dict(matched, strict=False)
        matched_params = sum(t.numel() for t in matched.values())
        total_params = sum(t.numel() for t in own.values())
        ckpt_params = sum(t.numel() for t in state.values() if torch.is_tensor(t))
        self._pretrained_load_report = {
            "path": str(path),
            "loaded": bool(matched),
            "matched_keys": len(matched),
            "checkpoint_keys": len(state),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "ignored_keys": len(ignored),
            "matched_params": matched_params,
            "backbone_params": total_params,
            "checkpoint_params": ckpt_params,
            "matched_backbone_ratio": matched_params / max(1, total_params),
            "matched_checkpoint_ratio": matched_params / max(1, ckpt_params),
        }
        print("[Pretrained Loading - TopoMamba3D]")
        print(f" - path: {self._pretrained_load_report['path']}")
        print(f" - backbone_params: {total_params:,} ({total_params / 1e6:.2f}M)")
        print(f" - actually_loaded_params: {matched_params:,} ({matched_params / 1e6:.2f}M)")
        print(f" - source_weight_params: {ckpt_params:,} ({ckpt_params / 1e6:.2f}M)")
        print(f" - mapped_keys: {len(matched)}/{len(state)}")
        print(f" - loading_ratio(params): {self._pretrained_load_report['matched_backbone_ratio'] * 100.0:.2f}%")
        return self._pretrained_load_report

    def forward(self, x):
        input_shape = x.shape[2:]
        x = self._adapt_input(x)
        logits4 = self.backbone(x)
        logits4 = self.topology_adapter(logits4)
        logits = self.output_head(logits4)
        if logits.shape[2:] != input_shape:
            logits = F.interpolate(logits, size=input_shape, mode="trilinear", align_corners=False)
        return logits


@register_model
def TopoMamba_3D_t(**kwargs):
    return TopoMamba3D(**kwargs)


@register_model
def topomamba_3d_t(**kwargs):
    return TopoMamba_3D_t(**kwargs)
