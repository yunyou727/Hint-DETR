# ------------------------------------------------------------------------
# Modified from DAB-DETR (https://github.com/IDEA-Research/DAB-DETR)
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
import math
import copy
import os
from functools import partial
from typing import Optional, List

from timm.layers import DropPath

from util.misc import inverse_sigmoid

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .attention import MultiheadAttention
import torch.utils.checkpoint as cp
from functools import partial, reduce
from operator import mul


# from .ops.modules import MSDeformAttn


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def gen_sineembed_for_position(pos_tensor, d_model=256):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(d_model // 2, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / (d_model // 2))
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_queries=300, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=4,
                 keep_query_pos=False, query_scale_type='cond_elewise',
                 num_patterns=0,
                 modulate_hw_attn=True,
                 bbox_embed_diff_each_layer=False,
                 num_prompt_tokens=8,
                 num_tokens=300,
                 num_feature_levels=4,
                 num_classes=7,
                 init_size=(64, 64)
                 ):

        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, keep_query_pos=keep_query_pos)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model, query_dim=query_dim, keep_query_pos=keep_query_pos,
                                          query_scale_type=query_scale_type,
                                          modulate_hw_attn=modulate_hw_attn,
                                          bbox_embed_diff_each_layer=bbox_embed_diff_each_layer)

        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, d_model)

        self.num_tokens = num_tokens
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.num_prompt_tokens = num_prompt_tokens
        self.num_encoder_layers = num_encoder_layers

        self.proj_layers = _get_clones(nn.Linear(num_tokens, num_prompt_tokens), self.num_encoder_layers)
        self.patch_proj_layers = _get_clones(nn.Linear(num_tokens, num_prompt_tokens), self.num_encoder_layers)
        self.mutil_proj = nn.Linear(num_tokens * num_classes, num_tokens)

        self.batch_norm = torch.nn.BatchNorm1d(num_tokens * d_model)
        self.src_flatten_norm = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, refpoint_embed, pos_embed, srcs, masks, pos_ad,
                i_ofpc_prompted_embed=None, pc_prompt_embed=None,
                en_if_prompt=False, de_if_prompt=False, if_pc_prompt=False, if_patch=False
                ):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed1 = pos_embed.flatten(2).permute(2, 0, 1)
        refpoint_embed = refpoint_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        if if_patch:
            i_ofpc_prompted_embed = i_ofpc_prompted_embed.unsqueeze(1).repeat(1, bs, 1).to(refpoint_embed.device)
            # mutil = torch.zeros(self.num_tokens, bs, self.d_model, device=refpoint_embed.device)
            mutil = i_ofpc_prompted_embed
        else:
            mutil = pc_prompt_embed.flatten(0, 1)
            mutil = self.mutil_proj(torch.transpose(mutil, 0, 1))
            mutil = torch.transpose(mutil, 0, 1)
            mutil = mutil.unsqueeze(1).repeat(1, bs, 1).to(refpoint_embed.device)
            # mutil = torch.zeros(self.num_tokens, bs, self.d_model, device=refpoint_embed.device)
        for lvl, (src_ad, mask_ad, pos_embed) in enumerate(zip(srcs, masks, pos_ad)):
            if lvl != 0:
                bs, c, h, w = src_ad.shape
                src_ad = src_ad.flatten(2).transpose(1, 2)  # bs, hw, c
                src_ad = torch.transpose(src_ad, 0, 1)
                mask_ad = mask_ad.flatten(1)  # bs, hw
                pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
                pos_embed = torch.transpose(pos_embed, 0, 1)
                if lvl != 1:
                    src_ad = self.encoder(src_ad, src_key_padding_mask=mask_ad, pos=pos_embed)
                pos_embed = torch.transpose(pos_embed, 0, 1)
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
                lvl_pos_embed = torch.transpose(lvl_pos_embed, 0, 1)
                lvl_pos_embed_flatten.append(lvl_pos_embed)
                src_flatten.append(src_ad)
                mask_flatten.append(mask_ad)
                if if_patch:
                    i_ofpc_prompted_embed = self.decoder(i_ofpc_prompted_embed, src_ad, memory_key_padding_mask=mask_ad,
                                                         pos=lvl_pos_embed, if_fpn=True)
                mutil = self.decoder(mutil, src_ad, memory_key_padding_mask=mask_ad,
                                     pos=lvl_pos_embed, if_fpn=True)

        p_mutil = torch.transpose(mutil, 0, 1)
        bs = p_mutil.shape[0]
        n = p_mutil.shape[1]
        dim = p_mutil.shape[2]
        if bs > 1:
            p_mutil = self.batch_norm(p_mutil.flatten(1, 2)).reshape(bs, n, dim)
        p_mutil = torch.sum(p_mutil, dim=0)
        p_mutil = p_mutil / bs  # (300,256)

        if if_patch:
            # edp = self.pe_deep_prompt_embeddings
            pp = torch.transpose(p_mutil, 0, 1)
            ppp = []
            for layer_id, layer in enumerate(self.patch_proj_layers):
                ppp.append(torch.transpose(layer(pp), 0, 1))
            edp = torch.stack(ppp, 0)

        else:
            ppe = torch.transpose(p_mutil, 0, 1)
            pppe = []
            for layer_id, layer in enumerate(self.proj_layers):
                pppe.append(torch.transpose(layer(ppe), 0, 1))
            edp = torch.stack(pppe, 0)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed1,
                              prompt_embeddings=edp,
                              if_prompt=en_if_prompt
                              )

        num_queries = refpoint_embed.shape[0]
        if not if_patch:
            if self.num_patterns == 0:
                tgt = torch.zeros(num_queries, bs, self.d_model, device=refpoint_embed.device)  # gaicheng ones
            else:
                tgt = self.patterns.weight[:, None, None, :].repeat(1, self.num_queries, bs, 1).flatten(0,
                                                                                                        1)  # n_q*n_pat, bs, d_model
                refpoint_embed = refpoint_embed.repeat(self.num_patterns, 1, 1)  # n_q*n_pat, bs, d_model

        else:
            tgt = i_ofpc_prompted_embed

        if if_patch:
            pp = torch.transpose(p_mutil, 0, 1)
            ppp = []
            for layer_id, layer in enumerate(self.patch_proj_layers):
                ppp.append(torch.transpose(layer(pp), 0, 1))
            ddp = torch.stack(ppp, 0)

        else:
            ppd = torch.transpose(p_mutil, 0, 1)
            pppd = []
            for layer_id, layer in enumerate(self.proj_layers):
                pppd.append(torch.transpose(layer(ppd), 0, 1))
            ddp = torch.stack(pppd, 0)
        hs, references = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                      pos=pos_embed1, refpoints_unsigmoid=refpoint_embed,
                                      prompt_embeddings=ddp,
                                      if_prompt=de_if_prompt,
                                      pc_prompt_embed=pc_prompt_embed,
                                      if_pc_prompt=if_pc_prompt,
                                      if_patch=if_patch
                                      )
        return hs, references


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, d_model=256):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                prompt_embeddings=None,
                if_prompt=False,
                ):
        output = src

        for layer_id, layer in enumerate(self.layers):
            # rescale the content and pos sim
            pos_scales = self.query_scale(output)
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos * pos_scales,
                           i_ofprompt_embeddings=prompt_embeddings[
                               layer_id] if prompt_embeddings is not None else None,
                           if_prompt=if_prompt
                           )
        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False,
                 d_model=256, query_dim=2, keep_query_pos=False, query_scale_type='cond_elewise',
                 modulate_hw_attn=False,
                 bbox_embed_diff_each_layer=False,
                 ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate
        self.query_dim = query_dim

        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        self.query_scale_type = query_scale_type
        if query_scale_type == 'cond_elewise':
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale = MLP(d_model, d_model, 1, 2)
        elif query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(num_layers, d_model)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)

        self.bbox_embed = None
        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer

        if modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)

        if not keep_query_pos:
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
                if_fpn=False,
                prompt_embeddings=None,
                if_prompt=False,
                pc_prompt_embed=None,
                if_pc_prompt=False, if_patch=False
                ):
        output = tgt

        if if_fpn:
            for layer_id, layer in enumerate(self.layers):
                output = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos,
                               is_first=(layer_id == 0),
                               if_fpn=if_fpn)

            if self.norm is not None:
                output = self.norm(output)

            return output
        intermediate = []
        # 将 x,y,w,h 缩放到 0~1
        reference_points = refpoints_unsigmoid.sigmoid()
        # 收集每层的参考点，除第一层外，每层均会进行校正
        ref_points = [reference_points]

        # import ipdb; ipdb.set_trace()

        if if_patch:
            for layer_id, layer in enumerate(self.layers):

                output = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos,
                               is_first=(layer_id == 0),
                               i_ofprompt_embeddings=prompt_embeddings[layer_id], if_prompt=if_prompt,
                               if_patch=if_patch
                               )
                if self.return_intermediate:
                    intermediate.append(self.norm(output))

        else:
            for layer_id, layer in enumerate(self.layers):
                # (num_queries, batch_size, 4)
                obj_center = reference_points[..., :self.query_dim]  # [num_queries, batch_size, 2]
                query_sine_embed = gen_sineembed_for_position(obj_center, self.d_model)
                query_pos = self.ref_point_head(query_sine_embed)

                if self.query_scale_type != 'fix_elewise':
                    if layer_id == 0:
                        pos_transformation = 1
                    else:
                        pos_transformation = self.query_scale(output)
                else:
                    pos_transformation = self.query_scale.weight[layer_id]

                query_sine_embed = query_sine_embed[..., :self.d_model] * pos_transformation

                # modulated HW attentions
                if self.modulate_hw_attn:
                    # (num_queries,bs,2)
                    refHW_cond = self.ref_anchor_head(output).sigmoid()  # nq, bs, 2
                    query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / obj_center[..., 2]).unsqueeze(-1)
                    query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / obj_center[..., 3]).unsqueeze(-1)

                if if_pc_prompt and layer_id == 0:
                    output = layer(output, pc_prompt_embed, if_pc_prompt=True)

                output = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                               is_first=(layer_id == 0),
                               i_ofprompt_embeddings=prompt_embeddings[layer_id], if_prompt=if_prompt,
                               if_patch=if_patch)

                # iter update
                if self.bbox_embed is not None:
                    if self.bbox_embed_diff_each_layer:
                        tmp = self.bbox_embed[layer_id](output)
                    else:
                        tmp = self.bbox_embed(output)
                    tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
                    new_reference_points = tmp[..., :self.query_dim].sigmoid()
                    if layer_id != self.num_layers - 1:
                        ref_points.append(new_reference_points)

                    reference_points = new_reference_points.detach()

                if self.return_intermediate:
                    intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    # (num_layers,bs,num_queries,d_model)
                    torch.stack(intermediate).transpose(1, 2),
                    # (num_layers,bs,num_queries,4)
                    torch.stack(ref_points).transpose(1, 2),
                ]
            else:
                return [
                    # (num_layers,bs,num_queries,d_model)
                    torch.stack(intermediate).transpose(1, 2),
                    # (1,bs,num_queries,4)
                    reference_points.unsqueeze(0).transpose(1, 2)
                ]
        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, dim_feedforward_1=2048):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.prompt_dropout = nn.Dropout(dropout)

        self.q_linear1 = nn.Linear(d_model, dim_feedforward_1)
        self.q_linear2 = nn.Linear(dim_feedforward_1, d_model)
        self.q_activation = _get_activation_fn(activation)
        self.q_dropout1 = nn.Dropout(dropout)
        self.q_dropout2 = nn.Dropout(dropout)
        self.q_norm = nn.LayerNorm(d_model)

        self.kv_linear1 = nn.Linear(d_model, dim_feedforward_1)
        self.kv_linear2 = nn.Linear(dim_feedforward_1, d_model)
        self.kv_activation = _get_activation_fn(activation)
        self.kv_dropout1 = nn.Dropout(dropout)
        self.kv_dropout2 = nn.Dropout(dropout)
        self.kv_norm = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def incorporate_prompt(self, x, i_ofprompt_embeddings):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[1]
        x = torch.cat((
            # x[:0, :, :],
            self.prompt_dropout(i_ofprompt_embeddings.unsqueeze(1).repeat(1, B, 1)),
            x[0:, :, :]
        ), dim=0)
        # (batch_size,  n_prompt + n_patches, hidden_dim)
        return x

    def k_proj(self, x):
        x1 = self.q_linear2(self.q_dropout1(self.q_activation(self.q_linear1(x))))
        x = x + self.q_dropout2(x1)
        x = self.q_norm(x)
        return x

    def v_proj(self, x):
        x1 = self.kv_linear2(self.kv_dropout1(self.kv_activation(self.kv_linear1(x))))
        x = x + self.kv_dropout2(x1)
        x = self.kv_norm(x)
        return x

    def prompt_forward(self,
                       src,
                       src_mask: Optional[Tensor] = None,
                       src_key_padding_mask: Optional[Tensor] = None,
                       pos: Optional[Tensor] = None, i_ofprompt_embeddings=None):

        q = k = self.with_pos_embed(src, pos)
        # n1 = src.shape[0]  # 初始向量个数494
        n2 = i_ofprompt_embeddings.shape[0]  # vpt的num token数
        k = self.incorporate_prompt(k, self.k_proj(i_ofprompt_embeddings))
        v = self.incorporate_prompt(src, self.v_proj(i_ofprompt_embeddings))

        skpm0 = torch.zeros(src.shape[1], n2, dtype=torch.bool)  # [bs,num_token]
        device = src_key_padding_mask.device
        skpm0 = skpm0.to(device)

        src_key_padding_mask1 = torch.cat((skpm0, src_key_padding_mask), dim=1)

        src2 = self.self_attn(q, k, value=v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask1)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                i_ofprompt_embeddings=None,
                if_prompt=False):
        if if_prompt:
            return self.prompt_forward(src, src_mask, src_key_padding_mask, pos, i_ofprompt_embeddings)
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, keep_query_pos=False,
                 rm_self_attn_decoder=False, dim_feedforward_1=2048):
        super().__init__()
        # Decoder Self-Attention
        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.keep_query_pos = keep_query_pos

        self.fpn_q = nn.Linear(d_model, 2 * d_model)
        self.fpn_k = nn.Linear(d_model, 2 * d_model)

        self.prompt_dropout = nn.Dropout(dropout)
        self.q_linear1 = nn.Linear(d_model, dim_feedforward_1)
        self.q_linear2 = nn.Linear(dim_feedforward_1, d_model)
        self.q_activation = _get_activation_fn(activation)
        self.q_dropout1 = nn.Dropout(dropout)
        self.q_dropout2 = nn.Dropout(dropout)
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_linear1 = nn.Linear(d_model, dim_feedforward_1)
        self.kv_linear2 = nn.Linear(dim_feedforward_1, d_model)
        self.kv_activation = _get_activation_fn(activation)
        self.kv_dropout1 = nn.Dropout(dropout)
        self.kv_dropout2 = nn.Dropout(dropout)
        self.kv_norm = nn.LayerNorm(d_model)
        self.pc_norm = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def fpn_forward(self, tgt, memory, memory_key_padding_mask, pos, is_first=False):
        if not self.rm_self_attn_decoder:
            q_content = self.sa_qcontent_proj(tgt)
            k_content = self.sa_kcontent_proj(tgt)
            v = self.sa_v_proj(tgt)
            num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape
            q = q_content
            k = k_content
            tgt2 = self.self_attn(q, k, value=v)[0]

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we concatenate the positional embedding predicted from
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first or self.keep_query_pos:
            q = q_content
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = self.fpn_q(q)
        k = self.fpn_k(k)

        tgt2 = self.cross_attn(query=q,
                               key=k,
                               value=v,
                               key_padding_mask=memory_key_padding_mask)[0]
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def incorporate_prompt(self, x, i_ofprompt_embeddings):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[1]
        x = torch.cat((
            # x[:0, :, :],
            self.prompt_dropout(i_ofprompt_embeddings.unsqueeze(1).repeat(1, B, 1)),
            x[0:, :, :]
        ), dim=0)
        # (batch_size,  n_prompt + n_patches, hidden_dim)
        return x

    def k_proj(self, x):
        x1 = self.q_linear2(self.q_dropout1(self.q_activation(self.q_linear1(x))))
        x = x + self.q_dropout2(x1)
        x = self.q_norm(x)
        return x

    def v_proj(self, x):
        x1 = self.kv_linear2(self.kv_dropout1(self.kv_activation(self.kv_linear1(x))))
        x = x + self.kv_dropout2(x1)
        x = self.kv_norm(x)
        return x

    def prompt_forward(
            self, tgt, memory,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            query_pos: Optional[Tensor] = None,
            query_sine_embed=None,
            is_first=False,
            i_ofprompt_embeddings=None
    ):
        if not self.rm_self_attn_decoder:
            q_content = self.sa_qcontent_proj(
                tgt)  # target is the input of the first decoder layer, zero by default.
            q_pos = self.sa_qpos_proj(query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)
            q = q_content + q_pos
            k = k_content + k_pos

            # 取0是拿出注意力施加在 value 后的结果，1是注意力权重矩阵
            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            # ========== End of Self-Attention =============

            # 残差连接
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)
        # 第一层由于没有足够的位置信息，因此默认要加上位置部分
        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model // self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        n2 = i_ofprompt_embeddings.shape[0]  # vpt的num token数
        k = self.incorporate_prompt(k, self.fpn_k(self.k_proj(i_ofprompt_embeddings)))
        v = self.incorporate_prompt(v, self.v_proj(i_ofprompt_embeddings))
        skpm0 = torch.zeros(memory.shape[1], n2, dtype=torch.bool)  # [bs,num_token]
        device = memory_key_padding_mask.device
        skpm0 = skpm0.to(device)
        memory_key_padding_mask1 = torch.cat((skpm0, memory_key_padding_mask), dim=1)  # 避免新加入token的被mask屏蔽

        # 取0是拿出注意力施加在 value 后的结果，1是注意力权重矩阵
        tgt2 = self.cross_attn(
            query=q,
            key=k,
            value=v, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask1
        )[0]
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        # linear->activation->dropout->linear
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))

        # dropout->residual->norm
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

    def patch_forward(
            self, tgt, memory,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            i_ofprompt_embeddings=None,
    ):
        if not self.rm_self_attn_decoder:
            q_content = self.sa_qcontent_proj(
                tgt)
            k_content = self.sa_kcontent_proj(tgt)
            q = q_content
            k = k_content
            v = self.sa_v_proj(tgt)
            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
        # ========== Begin of Cross-Attention =============
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)
        hw, _, _ = k_content.shape
        k_pos = self.ca_kpos_proj(pos)
        q = q_content
        k = k_content + k_pos
        n2 = i_ofprompt_embeddings.shape[0]  # vpt的num token数
        k = self.incorporate_prompt(k, self.k_proj(i_ofprompt_embeddings))
        v = self.incorporate_prompt(v, self.v_proj(i_ofprompt_embeddings))
        skpm0 = torch.zeros(memory.shape[1], n2, dtype=torch.bool)  # [bs,num_token]
        device = memory_key_padding_mask.device
        skpm0 = skpm0.to(device)
        memory_key_padding_mask1 = torch.cat((skpm0, memory_key_padding_mask), dim=1)

        q = self.fpn_q(
            q)
        k = self.fpn_k(k)
        tgt2 = self.cross_attn(
            query=q,
            key=k,
            value=v, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask1
        )[0]
        # ========== End of Cross-Attention =============
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def pc_forward_post(self, tgt, memory):
        bs = tgt.shape[1]
        memory = self.pc_norm(memory.flatten(0, 1).unsqueeze(1).repeat(1, bs, 1))
        q = self.ca_qcontent_proj(tgt)
        k = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)
        q = self.fpn_q(q)
        k = self.fpn_k(k)
        # 取0是拿出注意力施加在 value 后的结果，1是注意力权重矩阵
        tgt2 = self.cross_attn(
            query=q,
            key=k,
            value=v
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed=None,
                is_first=False,
                if_fpn=False,
                i_ofprompt_embeddings=None,
                if_prompt=False,
                if_pc_prompt=False, if_patch=False
                ):
        if if_fpn:
            return self.fpn_forward(tgt, memory, memory_key_padding_mask, pos, is_first=is_first)

        if if_patch:
            return self.patch_forward(tgt, memory, tgt_mask, memory_mask,
                                      tgt_key_padding_mask, memory_key_padding_mask, pos,
                                      i_ofprompt_embeddings=i_ofprompt_embeddings, )
        if if_pc_prompt:
            return self.pc_forward_post(tgt, memory)

        if if_prompt:
            return self.prompt_forward(tgt, memory, tgt_mask, memory_mask,
                                       tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos,
                                       query_sine_embed, is_first,
                                       i_ofprompt_embeddings=i_ofprompt_embeddings)

        # ========== Begin of Self-Attention =============
        if not self.rm_self_attn_decoder:
            # Apply projections here
            # shape: num_queries x batch_size x 256
            q_content = self.sa_qcontent_proj(tgt)  # target is the input of the first decoder layer. zero by default.
            q_pos = self.sa_qpos_proj(query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)

            num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape

            q = q_content + q_pos
            k = k_content + k_pos

            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            # ========== End of Self-Attention =============

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we concatenate the positional embedding predicted from
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model // self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(query=q,
                               key=k,
                               value=v, attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=4,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
