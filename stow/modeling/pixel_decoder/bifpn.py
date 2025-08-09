import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY


class SeparableConv2d(nn.Module):
    """Depthwise Separable Conv = depthwise + pointwise"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm: Optional[str] = None,
        activation: bool = True,
    ):
        super().__init__()
        self.depthwise = Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=not norm,
            norm=get_norm(norm, in_channels),
        )
        self.pointwise = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=not norm,
            norm=get_norm(norm, out_channels),
        )
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(x)


class BiFPNLayer(nn.Module):
    """单级 BiFPN 融合层"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = "GN",
    ):
        super().__init__()
        self.out_channels = out_channels
        # 两个可学习的权重（fast normalized fusion）
        self.w1 = nn.Parameter(torch.ones(2))
        self.w2 = nn.Parameter(torch.ones(3))

        # 输入 1×1 投影卷积
        self.p5_in = Conv2d(in_channels, out_channels, 1)  # P5
        self.p4_in = Conv2d(in_channels, out_channels, 1)  # P4
        self.p3_in = Conv2d(in_channels, out_channels, 1)  # P3
        self.p2_in = Conv2d(in_channels, out_channels, 1)  # P2

        # 融合后卷积
        self.p4_up_conv = SeparableConv2d(out_channels, out_channels, norm=norm)
        self.p3_up_conv = SeparableConv2d(out_channels, out_channels, norm=norm)
        self.p2_up_conv = SeparableConv2d(out_channels, out_channels, norm=norm)

        self.p3_down_conv = SeparableConv2d(out_channels, out_channels, norm=norm)
        self.p4_down_conv = SeparableConv2d(out_channels, out_channels, norm=norm)
        self.p5_down_conv = SeparableConv2d(out_channels, out_channels, norm=norm)

    def _resize(self, x: torch.Tensor, size: torch.Size) -> torch.Tensor:
        return F.interpolate(x, size=size, mode="nearest")

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        keys = list(features.keys())
        c2, c3, c4, c5 = [features[k] for k in keys]

        # 1×1 投影
        p2, p3, p4, p5 = self.p2_in(c2), self.p3_in(c3), self.p4_in(c4), self.p5_in(c5)

        # --- 自顶向下 ---
        # P4_td = w1 * P4 + w2 * resize(P5)
        w1 = F.relu(self.w1)
        p4_td = (w1[0] * p4 + w1[1] * self._resize(p5, p4.shape[-2:])) / (w1.sum() + 1e-4)
        p4_td = self.p4_up_conv(p4_td)

        # P3_td = w1 * P3 + w2 * resize(P4_td)
        p3_td = (w1[0] * p3 + w1[1] * self._resize(p4_td, p3.shape[-2:])) / (w1.sum() + 1e-4)
        p3_td = self.p3_up_conv(p3_td)

        # P2_out = w1 * P2 + w2 * resize(P3_td)
        p2_out = (w1[0] * p2 + w1[1] * self._resize(p3_td, p2.shape[-2:])) / (w1.sum() + 1e-4)
        p2_out = self.p2_up_conv(p2_out)

        # --- 自底向上 ---
        # P3_out = w1 * P3_td + w2 * resize(P4_td) + w3 * P3
        w2 = F.relu(self.w2)
        p3_out = (
            w2[0] * p3_td
            + w2[1] * F.interpolate(p2_out, size=p3_td.shape[-2:], mode="nearest")
            + w2[2] * p3
        ) / (w2.sum() + 1e-4)
        p3_out = self.p3_down_conv(p3_out)

        p4_out = (
            w2[0] * p4_td
            + w2[1] * F.interpolate(p3_out, size=p4_td.shape[-2:], mode="nearest")
            + w2[2] * p4
        ) / (w2.sum() + 1e-4)
        p4_out = self.p4_down_conv(p4_out)

        p5_out = (
            w2[0] * p5
            + w2[1] * F.interpolate(p4_out, size=p5.shape[-2:], mode="nearest")
            + w2[2] * p5
        ) / (w2.sum() + 1e-4)
        p5_out = self.p5_down_conv(p5_out)

        return {
            keys[0]: p2_out,
            keys[1]: p3_out,
            keys[2]: p4_out,
            keys[3]: p5_out,
        }


class BiFPN(nn.Module):
    """
    完整的 BiFPN，可级联多层
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        out_channels: int = 256,
        num_layers: int = 3,
        norm: str = "GN",
    ):
        super().__init__()
        self.out_channels = out_channels

        # 1. 为每个 level 创建 1×1 投影
        self.input_proj = nn.ModuleDict()
        for name, spec in input_shape.items():
            self.input_proj[name] = Conv2d(
                spec.channels,
                out_channels,
                kernel_size=1,
                bias=False,
                norm=get_norm(norm, out_channels),
            )

        # 2. 级联 BiFPN 层
        self.layers = nn.ModuleList(
            [
                BiFPNLayer(out_channels, out_channels, norm=norm)
                for _ in range(num_layers)
            ]
        )

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        """
        为不同通道的特征创建 1×1 卷积对齐，不再要求 backbone 每层通道一致
        """
        return {
            "input_shape": input_shape,
            "out_channels": cfg.MODEL.BiFPN.OUT_CHANNELS,
            "num_layers": cfg.MODEL.BiFPN.NUM_LAYERS,
            "norm": cfg.MODEL.BiFPN.NORM,
        }

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 先投影到统一通道
        projected = {k: self.input_proj[k](v) for k, v in features.items()}
        # 再级联 BiFPN
        for layer in self.layers:
            projected = layer(projected)
        return projected
    
    def forward_features(self, features: Dict[str, torch.Tensor]):
        feats = self.forward(features)
        keys = sorted(feats.keys(), key=lambda k: feats[k].shape[-1], reverse=True)
        # 只取 3 层（例如 res5/res4/res3）
        multi_scale_features = [feats[k] for k in keys[:3]]
        mask_features = multi_scale_features[-1]   # res3
        return mask_features, None, multi_scale_features


# ========== 注册到 Detectron2 ==========
SEM_SEG_HEADS_REGISTRY.register()(BiFPN)