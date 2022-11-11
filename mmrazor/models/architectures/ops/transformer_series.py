# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch
import torch.nn as nn
from mmengine.model.weight_init import trunc_normal_


class RelativePosition2D(nn.Module):
    """Rethinking and Improving Relative Position Encoding for Vision
    Transformer.

    ICCV 2021. https://arxiv.org/pdf/2107.14222.pdf
    Image RPE (iRPE for short) methods are new relative position encoding
    methods dedicated to 2D images.
    Args:
        head_dims (int): embedding dims of relative position.
        max_relative_position (int): The max relative position distance.
    """

    def __init__(self, head_dims: int, max_relative_position: int = 14):
        super().__init__()

        self.head_dims = head_dims
        self.max_relative_position = max_relative_position
        # The first element in embeddings_table_v is the vertical embedding
        # for the class
        self.embeddings_table_v = nn.Parameter(
            torch.randn(max_relative_position * 2 + 2, head_dims))
        self.embeddings_table_h = nn.Parameter(
            torch.randn(max_relative_position * 2 + 2, head_dims))

        trunc_normal_(self.embeddings_table_v, std=.02)
        trunc_normal_(self.embeddings_table_h, std=.02)

    def forward(self, length_q, length_k):
        # remove the first cls token distance computation
        length_q = length_q - 1
        length_k = length_k - 1
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        # compute the row and column distance
        distance_mat_v = (
            range_vec_k[None, :] // int(length_q**0.5) -
            range_vec_q[:, None] // int(length_q**0.5))
        distance_mat_h = (
            range_vec_k[None, :] % int(length_q**0.5) -
            range_vec_q[:, None] % int(length_q**0.5))
        # clip the distance to the range of
        # [-max_relative_position, max_relative_position]
        distance_mat_clipped_v = torch.clamp(distance_mat_v,
                                             -self.max_relative_position,
                                             self.max_relative_position)
        distance_mat_clipped_h = torch.clamp(distance_mat_h,
                                             -self.max_relative_position,
                                             self.max_relative_position)

        # translate the distance from [1, 2 * max_relative_position + 1],
        # 0 is for the cls token
        final_mat_v = distance_mat_clipped_v + self.max_relative_position + 1
        final_mat_h = distance_mat_clipped_h + self.max_relative_position + 1
        # pad the 0 which represent the cls token
        final_mat_v = torch.nn.functional.pad(final_mat_v, (1, 0, 1, 0),
                                              'constant', 0)
        final_mat_h = torch.nn.functional.pad(final_mat_h, (1, 0, 1, 0),
                                              'constant', 0)

        final_mat_v = torch.LongTensor(final_mat_v)
        final_mat_h = torch.LongTensor(final_mat_h)
        # get the embeddings with the corresponding distance
        embeddings = self.embeddings_table_v[
            final_mat_v] + self.embeddings_table_h[final_mat_h]

        return embeddings


class MultiheadAttention(nn.Module):
    """Multi-head Attention Module with iRPE.

    This module implements multi-head attention that supports different input
    dims and embed dims. And it also supports a shortcut from ``value``, which
    is useful if input dims is not the same with embed dims.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        dropout_layer (dict): The dropout config before adding the shortcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        relative_position (bool, optional): Whether use relative position.
            Defaults to True.
        max_relative_position (int): The max relative position distance.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        v_shortcut (bool): Add a shortcut from value to output. It's usually
            used if ``input_dims`` is different from ``embed_dims``.
            Defaults to False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 input_dims: Optional[int] = None,
                 attn_drop_rate: float = 0.,
                 proj_drop: float = 0.,
                 dropout_layer: Dict = dict(type='Dropout', drop_prob=0.),
                 relative_position: Optional[bool] = True,
                 max_relative_position: int = 14,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 proj_bias: bool = True,
                 v_shortcut: bool = False,
                 init_cfg: Optional[dict] = None):
        super().__init__()

        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut
        self.relative_position = relative_position
        self.max_relative_position = max_relative_position
        self.attn_drop_rate = attn_drop_rate

        self.head_dims = 64  # unit
        self.scale = qk_scale or self.head_dims**-0.5

        self.q_embed_dims = num_heads * self.head_dims

        self.w_qs = nn.Linear(
            self.input_dims, num_heads * self.head_dims, bias=qkv_bias)
        self.w_ks = nn.Linear(
            self.input_dims, num_heads * self.head_dims, bias=qkv_bias)
        self.w_vs = nn.Linear(
            self.input_dims, num_heads * self.head_dims, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(
            num_heads * self.head_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.out_drop = nn.Dropout(dropout_layer['drop_prob'])

        # image relative position encoding
        if self.relative_position:
            self.rel_pos_embed_k = RelativePosition2D(
                self.head_dims, self.max_relative_position)
            self.rel_pos_embed_v = RelativePosition2D(
                self.head_dims, self.max_relative_position)

    def forward(self, x):
        B, N, _ = x.shape

        q = self.w_qs(x).view(B, N, self.num_heads, self.head_dims)
        k = self.w_ks(x).view(B, N, self.num_heads, self.head_dims)
        v = self.w_vs(x).view(B, N, self.num_heads, self.head_dims)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_position:
            r_p_k = self.rel_pos_embed_k(N, N)
            attn = attn + (q.permute(2, 0, 1, 3).reshape(N, self.num_heads * B, -1)  # noqa: E501
                           @ r_p_k.transpose(2, 1)) \
                .transpose(1, 0).reshape(B, self.num_heads, N, N) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        if self.relative_position:
            r_p_v = self.rel_pos_embed_v(N, N)
            t_attn = attn.permute(2, 0, 1, 3).reshape(N, B * self.num_heads,
                                                      -1)
            x = x + (t_attn @ r_p_v).transpose(1, 0).reshape(
                B, self.num_heads, N, -1).transpose(2, 1).reshape(B, N, -1)

        x = self.proj(x)
        x = self.out_drop(self.proj_drop(x))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x
