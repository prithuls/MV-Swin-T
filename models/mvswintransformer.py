# --------------------------------------------------------
# Multi View Swin Transformer
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Prithul Sarker and Sushmita Sarker
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_
from models.singleviewutils import PatchEmbed, PatchMerging, SwinTransformerBlock_singleview
from models.multiviewutils import OmniAttentionTransformerBlock_multiview

class BasicLayer_multiview(nn.Module):
    """ A basic Multi view Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0, diff_attn=True):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            OmniAttentionTransformerBlock_multiview(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (
                                     i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(
                                     drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size,
                                 diff_attn=diff_attn)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample_1 = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer)
            self.downsample_2 = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample_1 = None

    def forward(self, x1, x2):
        for blk in self.blocks:
            if self.use_checkpoint:
                x1, x2 = checkpoint.checkpoint(blk, x1, x2)
            else:
                x1, x2 = blk(x1, x2)
        if self.downsample_1 is not None:
            x1 = self.downsample_1(x1)
            x2 = self.downsample_2(x2)
        return x1, x2

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1_1.bias, 0)
            nn.init.constant_(blk.norm1_1.weight, 0)
            nn.init.constant_(blk.norm2_1.bias, 0)
            nn.init.constant_(blk.norm2_1.weight, 0)
            nn.init.constant_(blk.norm1_2.bias, 0)
            nn.init.constant_(blk.norm1_2.weight, 0)
            nn.init.constant_(blk.norm2_2.bias, 0)
            nn.init.constant_(blk.norm2_2.weight, 0)


class BasicLayer_singleview(nn.Module):
    """ A basic single view Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock_singleview(dim=dim, input_resolution=input_resolution,
                                          num_heads=num_heads, window_size=window_size,
                                          shift_size=0 if (
                                              i % 2 == 0) else window_size // 2,
                                          mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[i] if isinstance(
                                              drop_path, list) else drop_path,
                                          norm_layer=norm_layer,
                                          pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)



class MVSwinTransformer(nn.Module):
    r""" Multiview Swin Transformer
        A PyTorch impl of : `MV-Swin-T: Mammogram Classification with Multi-View Swin Transformer`
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 12 (for 384x384 input image)
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
        diff_attn_layers (list(int)): Layers to use dynamic attention mechanism.
    """

    def __init__(self, img_size=384, patch_size=4, in_chans=3, num_classes=1,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], 
                 diff_attn_layers = [1], **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed_1 = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.patch_embed_2 = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed_1.num_patches
        patches_resolution = self.patch_embed_1.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed_1 = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
            self.absolute_pos_embed_2 = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed_1, std=.02)
            trunc_normal_(self.absolute_pos_embed_2, std=.02)

        self.pos_drop_1 = nn.Dropout(p=drop_rate)
        self.pos_drop_2 = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule

        # build multiview layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers // 2):
            layer = BasicLayer_multiview(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(
                                   depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (
                                   i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer],
                               diff_attn=True if i_layer in diff_attn_layers else False)
            self.layers.append(layer)

        fused_in_dim = patches_resolution[0] // (2 **
                                                 i_layer) * window_size * 2
        fused_out_dim = fused_in_dim // 2
        self.fc_layer_fused = nn.Linear(fused_in_dim, fused_out_dim)

        # build fused layers
        self.layers_fused = nn.ModuleList()
        for i_layer in range(2, self.num_layers):
            layer = BasicLayer_singleview(dim=int(embed_dim * 2 ** i_layer),
                                        input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                          patches_resolution[1] // (2 ** i_layer)),
                                        depth=depths[i_layer],
                                        num_heads=num_heads[i_layer],
                                        window_size=window_size,
                                        mlp_ratio=self.mlp_ratio,
                                        qkv_bias=qkv_bias,
                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=dpr[sum(depths[:i_layer]):sum(
                                            depths[:i_layer + 1])],
                                        norm_layer=norm_layer,
                                        downsample=PatchMerging if (
                                            i_layer < self.num_layers - 1) else None,
                                        use_checkpoint=use_checkpoint,
                                        pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers_fused.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(
            self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.act = nn.Sigmoid()

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

        for bly in self.layers_fused:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward_features(self, x1, x2):
        x1 = self.patch_embed_1(x1)
        x2 = self.patch_embed_2(x2)
        if self.ape:
            x1 = x1 + self.absolute_pos_embed_1
            x2 = x2 + self.absolute_pos_embed_2
        x1 = self.pos_drop_1(x1)
        x2 = self.pos_drop_2(x2)

        for layer in self.layers:
            x1, x2 = layer(x1, x2)

        x = torch.cat((x1, x2), dim=1).permute(0, 2, 1)
        # print("fc_fuse: ", x1.shape, x2.shape, x.shape, int(self.embed_dim * 2 ** (self.num_layers - 3)), self.patches_resolution[0] // (2 ** 1), self.patches_resolution[1] // (2 ** 1))
        x = self.fc_layer_fused(x).permute(0, 2, 1)

        for layer in self.layers_fused:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x1, x2):
        x = self.forward_features(x1, x2)
        x = self.head(x)
        return self.act(x)
