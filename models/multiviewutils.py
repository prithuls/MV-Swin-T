# --------------------------------------------------------
# Multi View Swin Transformer
# Copyright (c) 2024
# Licensed under The MIT License [see LICENSE for details]
# Written by Prithul Sarker and Sushmita Sarker
# --------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple
import numpy as np
from .singleviewutils import Mlp, window_partition, window_reverse



class DynamicAttention_multiview(nn.Module):
    r""" Window based multi-head self and cross attention (W-MDA) module 
    for multi view mammograms with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
        diff_attn (bool, optional): If True, use dynamic attention for different views. Default: True
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0], diff_attn=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size 
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.diff_attn = diff_attn

        # For first view
        self.logit_scale_1 = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp_1 = nn.Sequential(nn.Linear(2, 512, bias=True),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h_1 = torch.arange(
            -(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w_1 = torch.arange(
            -(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table_1 = torch.stack(
            torch.meshgrid([relative_coords_h_1,
                            relative_coords_w_1])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table_1[:, :, :,
                                    0] /= (pretrained_window_size[0] - 1)
            relative_coords_table_1[:, :, :,
                                    1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table_1[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table_1[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table_1 *= 8  # normalize to -8, 8
        self.relative_coords_table_1 = (torch.sign(relative_coords_table_1) * torch.log2(
            torch.abs(relative_coords_table_1) + 1.0) / np.log2(8)).cuda()

        self.register_buffer("relative_coords_table",
                             self.relative_coords_table_1)

        # get pair-wise relative position index for each token inside the window
        coords_h_1 = torch.arange(self.window_size[0])
        coords_w_1 = torch.arange(self.window_size[1])
        coords_1 = torch.stack(torch.meshgrid(
            [coords_h_1, coords_w_1]))  
        coords_flatten_1 = torch.flatten(coords_1, 1) 
        
        relative_coords_1 = coords_flatten_1[:,
                                             :, None] - coords_flatten_1[:, None, :]
        relative_coords_1 = relative_coords_1.permute(
            1, 2, 0).contiguous()  
        relative_coords_1[:, :, 0] += self.window_size[0] - \
            1  
        relative_coords_1[:, :, 1] += self.window_size[1] - 1
        relative_coords_1[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index_1 = relative_coords_1.sum(-1)  
        self.register_buffer("relative_position_index_1",
                             relative_position_index_1)

        self.qkv_1 = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias_1 = nn.Parameter(torch.zeros(dim))
            self.v_bias_1 = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias_1 = None
            self.v_bias_1 = None
        self.attn_drop_1 = nn.Dropout(attn_drop)
        self.proj_1 = nn.Linear(dim, dim)
        self.proj_drop_1 = nn.Dropout(proj_drop)
        self.softmax_1 = nn.Softmax(dim=-1)

        # For second view

        self.logit_scale_2 = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp_2 = nn.Sequential(nn.Linear(2, 512, bias=True),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h_2 = torch.arange(
            -(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w_2 = torch.arange(
            -(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table_2 = torch.stack(
            torch.meshgrid([relative_coords_h_2,
                            relative_coords_w_2])).permute(1, 2, 0).contiguous().unsqueeze(0)  
        if pretrained_window_size[0] > 0:
            relative_coords_table_2[:, :, :,
                                    0] /= (pretrained_window_size[0] - 1)
            relative_coords_table_2[:, :, :,
                                    1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table_2[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table_2[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table_2 *= 8  # normalize to -8, 8
        self.relative_coords_table_2 = (torch.sign(relative_coords_table_2) * torch.log2(
            torch.abs(relative_coords_table_2) + 1.0) / np.log2(8)).cuda()

        self.register_buffer("relative_coords_table",
                             self.relative_coords_table_2)

        # get pair-wise relative position index for each token inside the window
        coords_h_2 = torch.arange(self.window_size[0])
        coords_w_2 = torch.arange(self.window_size[1])
        coords_2 = torch.stack(torch.meshgrid(
            [coords_h_2, coords_w_2])) 
        coords_flatten_2 = torch.flatten(coords_2, 1)  
        relative_coords_2 = coords_flatten_2[:,
                                             :, None] - coords_flatten_2[:, None, :]
        relative_coords_2 = relative_coords_2.permute(
            1, 2, 0).contiguous()  
        relative_coords_2[:, :, 0] += self.window_size[0] - \
            1  
        relative_coords_2[:, :, 1] += self.window_size[1] - 1
        relative_coords_2[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index_2 = relative_coords_2.sum(-1)  #
        self.register_buffer("relative_position_index_2",
                             relative_position_index_2)

        self.qkv_2 = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias_2 = nn.Parameter(torch.zeros(dim))
            self.v_bias_2 = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias_2 = None
            self.v_bias_2 = None
        self.attn_drop_2 = nn.Dropout(attn_drop)
        self.proj_2 = nn.Linear(dim, dim)
        self.proj_drop_2 = nn.Dropout(proj_drop)
        self.softmax_2 = nn.Softmax(dim=-1)

        # For concatanation
        if self.diff_attn:
            self.proj_concat_1 = nn.Linear(self.num_heads * 2, self.num_heads)
            self.proj_concat_2 = nn.Linear(self.num_heads * 2, self.num_heads)

    def forward(self, x1, x2, mask1=None, mask2=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x1.shape

        # First view
        qkv_bias_1 = None
        if self.q_bias_1 is not None:
            qkv_bias_1 = torch.cat((self.q_bias_1, torch.zeros_like(
                self.v_bias_1, requires_grad=False), self.v_bias_1))
        qkv_1 = F.linear(input=x1, weight=self.qkv_1.weight, bias=qkv_bias_1)
        qkv_1 = qkv_1.reshape(B_, N, 3, self.num_heads, -
                              1).permute(2, 0, 3, 1, 4)
        q_x1, k_x1, v_x1 = qkv_1[0], qkv_1[1], qkv_1[2]

        # Second view
        qkv_bias_2 = None
        if self.q_bias_2 is not None:
            qkv_bias_2 = torch.cat((self.q_bias_2, torch.zeros_like(
                self.v_bias_2, requires_grad=False), self.v_bias_2))
        qkv_2 = F.linear(input=x2, weight=self.qkv_2.weight, bias=qkv_bias_2)
        qkv_2 = qkv_2.reshape(B_, N, 3, self.num_heads, -
                              1).permute(2, 0, 3, 1, 4)
        q_x2, k_x2, v_x2 = qkv_2[0], qkv_2[1], qkv_2[2]

        if self.diff_attn:
            # cosine attention- first view
            attn_same_x1 = (F.normalize(q_x1, dim=-1) @
                            F.normalize(k_x1, dim=-1).transpose(-2, -1))
            attn_diff_x1 = (F.normalize(q_x1, dim=-1) @
                            F.normalize(k_x2, dim=-1).transpose(-2, -1))
            logit_scale_1 = torch.clamp(self.logit_scale_1, max=torch.log(
                torch.tensor(1. / 0.01).cuda())).exp()
            # print("before concat: ", attn_same_x1.shape, attn_diff_x1.shape)
            concat_attn_x1 = torch.concat(
                (attn_same_x1, attn_diff_x1), dim=1).permute(0, 2, 3, 1)
            # print("concat : ", concat_attn_x1.shape)

            ## For weighted average:
            # attn_x1 = (0.9 * attn_same_x1 + 0.1 * attn_diff_x1) * logit_scale_1

            ## For concatenation:
            attn_x1 = self.proj_concat_1(concat_attn_x1).permute(
                0, 3, 1, 2) * logit_scale_1

            # cosine attention- second view
            attn_same_x2 = (F.normalize(q_x2, dim=-1) @
                            F.normalize(k_x2, dim=-1).transpose(-2, -1))
            attn_diff_x2 = (F.normalize(q_x2, dim=-1) @
                            F.normalize(k_x1, dim=-1).transpose(-2, -1))
            logit_scale_2 = torch.clamp(self.logit_scale_2, max=torch.log(
                torch.tensor(1. / 0.01).cuda())).exp()
            ## For weighted average:
            # attn_x2 = (0.9 * attn_same_x2 + 0.1 * attn_diff_x2) * logit_scale_2

            ## For concatenation:
            concat_attn_x2 = torch.concat(
                (attn_same_x2, attn_diff_x2), dim=1).permute(0, 2, 3, 1)
            attn_x2 = self.proj_concat_2(concat_attn_x2).permute(
                0, 3, 1, 2) * logit_scale_2
        else:
            # cosine attention- first view
            attn_same_x1 = (F.normalize(q_x1, dim=-1) @
                            F.normalize(k_x1, dim=-1).transpose(-2, -1))
            logit_scale_1 = torch.clamp(self.logit_scale_1, max=torch.log(
                torch.tensor(1. / 0.01).cuda())).exp()
            attn_x1 = attn_same_x1 * logit_scale_1

            # cosine attention- second view
            attn_same_x2 = (F.normalize(q_x2, dim=-1) @
                            F.normalize(k_x2, dim=-1).transpose(-2, -1))
            logit_scale_2 = torch.clamp(self.logit_scale_2, max=torch.log(
                torch.tensor(1. / 0.01).cuda())).exp()
            attn_x2 = attn_same_x2 * logit_scale_2

        relative_position_bias_table_1 = self.cpb_mlp_1(
            self.relative_coords_table_1).view(-1, self.num_heads)
        relative_position_bias_1 = relative_position_bias_table_1[self.relative_position_index_1.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  
        relative_position_bias_1 = relative_position_bias_1.permute(
            2, 0, 1).contiguous()  
        relative_position_bias_1 = 16 * torch.sigmoid(relative_position_bias_1)
        attn_x1 = attn_x1 + relative_position_bias_1.unsqueeze(0)

        relative_position_bias_table_2 = self.cpb_mlp_2(
            self.relative_coords_table_2).view(-1, self.num_heads)
        relative_position_bias_2 = relative_position_bias_table_2[self.relative_position_index_2.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  
        relative_position_bias_2 = relative_position_bias_2.permute(
            2, 0, 1).contiguous()  
        relative_position_bias_2 = 16 * torch.sigmoid(relative_position_bias_2)
        attn_x2 = attn_x2 + relative_position_bias_2.unsqueeze(0)

        if mask1 is not None:
            nW = mask1.shape[0]
            attn_x1 = attn_x1.view(
                B_ // nW, nW, self.num_heads, N, N) + mask1.unsqueeze(1).unsqueeze(0)
            attn_x1 = attn_x1.view(-1, self.num_heads, N, N)
            attn_x1 = self.softmax_1(attn_x1)
        else:
            attn_x1 = self.softmax_1(attn_x1)

        if mask1 is not None:
            nW = mask1.shape[0]
            attn_x2 = attn_x2.view(
                B_ // nW, nW, self.num_heads, N, N) + mask2.unsqueeze(1).unsqueeze(0)
            attn_x2 = attn_x2.view(-1, self.num_heads, N, N)
            attn_x2 = self.softmax_2(attn_x2)
        else:
            attn_x2 = self.softmax_2(attn_x2)

        attn_x1 = self.attn_drop_1(attn_x1)
        x1 = (attn_x1 @ v_x1).transpose(1, 2).reshape(B_, N, C)
        x1 = self.proj_1(x1)
        x1 = self.proj_drop_1(x1)

        attn_x2 = self.attn_drop_2(attn_x2)
        x2 = (attn_x2 @ v_x2).transpose(1, 2).reshape(B_, N, C)
        x2 = self.proj_2(x2)
        x2 = self.proj_drop_2(x2)
        return x1, x2

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'


class OmniAttentionTransformerBlock_multiview(nn.Module):
    r""" Omni Attention Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
        diff_attn (bool, optional): If True, use dynamic attention for different views. Default: True
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=12, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.ReLU, norm_layer=nn.LayerNorm, pretrained_window_size=0, diff_attn=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1_1 = norm_layer(dim)
        self.norm1_2 = norm_layer(dim)
        self.attn = DynamicAttention_multiview(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size), diff_attn=diff_attn)

        self.drop_path_1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_1 = norm_layer(dim)
        self.norm2_2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_1 = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_2 = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1,
                                             self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x1, x2):
        H, W = self.input_resolution
        B, L, C = x1.shape
        assert L == H * W, "input feature has wrong size"

        shortcut_x1 = x1
        x1 = x1.view(B, H, W, C)
        shortcut_x2 = x2
        x2 = x2.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x1 = torch.roll(
                x1, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_x2 = torch.roll(
                x2, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x1 = x1
            shifted_x2 = x2

        x_windows_1 = window_partition(shifted_x1, self.window_size)
        x_windows_1 = x_windows_1.view(-1,
                                       self.window_size * self.window_size, C)

        x_windows_2 = window_partition(shifted_x2, self.window_size)
        x_windows_2 = x_windows_2.view(-1,
                                       self.window_size * self.window_size, C)

        # W-MDA/SW-MDA
        attn_windows_1, attn_windows_2 = self.attn(x_windows_1,
                                                   x_windows_2,
                                                   mask1=self.attn_mask,
                                                   mask2=self.attn_mask) 

        # merge windows
        attn_windows_1 = attn_windows_1.view(-1,
                                             self.window_size, self.window_size, C)
        shifted_x1 = window_reverse(
            attn_windows_1, self.window_size, H, W) 
        attn_windows_2 = attn_windows_2.view(-1,
                                             self.window_size, self.window_size, C)
        shifted_x2 = window_reverse(
            attn_windows_2, self.window_size, H, W) 

        # reverse cyclic shift
        if self.shift_size > 0:
            x1 = torch.roll(shifted_x1, shifts=(
                self.shift_size, self.shift_size), dims=(1, 2))
            x2 = torch.roll(shifted_x2, shifts=(
                self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x1 = shifted_x1
            x2 = shifted_x2
        x1 = x1.view(B, H * W, C)
        x1 = shortcut_x1 + self.drop_path_1(self.norm1_1(x1))
        x2 = x2.view(B, H * W, C)
        x2 = shortcut_x2 + self.drop_path_2(self.norm1_2(x2))

        # FFN
        x1 = x1 + self.drop_path_1(self.norm2_1(self.mlp_1(x1)))
        x2 = x2 + self.drop_path_2(self.norm2_2(self.mlp_2(x2)))

        return x1, x2