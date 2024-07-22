import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torch.utils.checkpoint as cp


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        try:
            ret = super().forward(x.type(torch.float32))
        except Exception as e:
            print(e)
        return ret.type(orig_type)


class GEN(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_dec_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, num_inter_dec_layrs=6,
                 return_intermediate_dec=False, num_queries=64, clip_dim=768, enable_cp=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, enable_cp)
        encoder_norm = LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        instance_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                         dropout, activation, normalize_before, False)
        instance_decoder_norm = LayerNorm(d_model)
        self.instance_decoder1 = TransformerDecoder(instance_decoder_layer,
                                                   num_dec_layers,
                                                   instance_decoder_norm,
                                                   return_intermediate=return_intermediate_dec)
        self.instance_decoder2 = TransformerDecoder(instance_decoder_layer,
                                                   num_dec_layers,
                                                   instance_decoder_norm,
                                                   return_intermediate=return_intermediate_dec)
        self.instance_decoder3 = TransformerDecoder(instance_decoder_layer,
                                                   num_dec_layers,
                                                   instance_decoder_norm,
                                                   return_intermediate=return_intermediate_dec)
        self.instance_decoder4 = TransformerDecoder(instance_decoder_layer,
                                                   num_dec_layers,
                                                   instance_decoder_norm,
                                                   return_intermediate=return_intermediate_dec)

        interaction_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                            dropout, activation, normalize_before, False)
        interaction_decoder_norm = LayerNorm(d_model)
        self.interaction_decoder = TransformerDecoder(interaction_decoder_layer,
                                                      num_dec_layers,
                                                      interaction_decoder_norm,
                                                      return_intermediate=return_intermediate_dec)

        clip_interaction_decoder_layer = TransformerDecoderFusionLayer(clip_dim, nhead, dim_feedforward,
                                                                       dropout, activation, normalize_before, enable_cp)
        clip_interaction_decoder_norm = LayerNorm(clip_dim)
        self.clip_interaction_decoder1 = TransformerDecoderCLIP(clip_interaction_decoder_layer,
                                                               num_dec_layers,
                                                               clip_interaction_decoder_norm,
                                                               return_intermediate=return_intermediate_dec)
        self.clip_interaction_decoder2 = TransformerDecoderCLIP(clip_interaction_decoder_layer,
                                                               num_dec_layers,
                                                               clip_interaction_decoder_norm,
                                                               return_intermediate=return_intermediate_dec)
        self.clip_interaction_decoder3 = TransformerDecoderCLIP(clip_interaction_decoder_layer,
                                                               num_dec_layers,
                                                               clip_interaction_decoder_norm,
                                                               return_intermediate=return_intermediate_dec)
        self.clip_interaction_decoder4 = TransformerDecoderCLIP(clip_interaction_decoder_layer,
                                                               num_dec_layers,
                                                               clip_interaction_decoder_norm,
                                                               return_intermediate=return_intermediate_dec)
        self.inter_guided_embedd = nn.Embedding(num_queries, clip_dim)
        self.queries2spacial_proj = nn.Linear(d_model, clip_dim)
        self.queries2spacial_proj_norm = LayerNorm(clip_dim)

        self.obj_class_fc = nn.Linear(d_model, clip_dim)
        self.obj_class_ln = LayerNorm(clip_dim)

        text_encoder_layer = nn.TransformerEncoderLayer(clip_dim, nhead, dim_feedforward, dropout, activation)
        self.text_encoder = nn.TransformerEncoder(text_encoder_layer, 3)

        self.text_fc = nn.Linear(clip_dim, d_model)
        self.text_ln = LayerNorm(d_model)
        self.text_fc2 = nn.Linear(clip_dim, d_model)
        self.text_ln2 = LayerNorm(d_model)
        self.text_fc3 = nn.Linear(clip_dim, d_model)
        self.text_ln3 = LayerNorm(d_model)
        self.text_fc4 = nn.Linear(clip_dim, d_model)
        self.text_ln4 = LayerNorm(d_model)

        self.inter_class_fc = nn.Linear(clip_dim+clip_dim+clip_dim+clip_dim, clip_dim)
        self.inter_class_ln = LayerNorm(clip_dim)

        self.hoi_cls = None

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.uniform_(self.inter_guided_embedd.weight)

    def forward(self, src, mask, query_embed_h, query_embed_o, pos_guided_embed, pos_embed, clip_model, clip_proj,
                clip_src, object_emb, object_mask, position_emb, position_mask, posture_emb, posture_mask,
                spatial_emb, spatial_mask):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        object_emb = object_emb.permute(1, 0, 2)
        position_emb = position_emb.permute(1, 0, 2)
        posture_emb = posture_emb.permute(1, 0, 2)
        spatial_emb = spatial_emb.permute(1, 0, 2)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        num_queries = query_embed_h.shape[0]

        query_embed_o = query_embed_o + pos_guided_embed
        query_embed_h = query_embed_h + pos_guided_embed
        query_embed_o = query_embed_o.unsqueeze(1).repeat(1, bs, 1)
        query_embed_h = query_embed_h.unsqueeze(1).repeat(1, bs, 1)
        ins_query_embed = torch.cat((query_embed_h, query_embed_o), dim=0)

        mask = mask.flatten(1)
        ins_tgt = torch.zeros_like(ins_query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        object_states = self.text_encoder(object_emb, src_key_padding_mask=object_mask)
        position_states = self.text_encoder(position_emb, src_key_padding_mask=position_mask)
        posture_states = self.text_encoder(posture_emb, src_key_padding_mask=posture_mask)
        spatial_states = self.text_encoder(spatial_emb, src_key_padding_mask=spatial_mask)
        object_states_sup = self.text_ln(self.text_fc(object_states))
        posture_states_sup = self.text_ln2(self.text_fc2(posture_states))
        position_states_sup = self.text_ln3(self.text_fc3(position_states))
        spatial_states_sup = self.text_ln4(self.text_fc4(spatial_states))

        # ins_hs = self.instance_decoder(ins_tgt, memory, memory_key_padding_mask=mask,
        #                                pos=pos_embed, query_pos=ins_query_embed)
        ins_hs1 = self.instance_decoder1(ins_tgt, memory, sup_memory=object_states_sup, sup_memory_mask=object_mask,
                                       memory_key_padding_mask=mask,
                                       pos=pos_embed, query_pos=ins_query_embed)
        ins_hs2 = self.instance_decoder2(ins_tgt, memory, sup_memory=posture_states_sup, sup_memory_mask=posture_mask,
                                       memory_key_padding_mask=mask,
                                       pos=pos_embed, query_pos=ins_query_embed)
        ins_hs3 = self.instance_decoder3(ins_tgt, memory, sup_memory=position_states_sup, sup_memory_mask=position_mask,
                                       memory_key_padding_mask=mask,
                                       pos=pos_embed, query_pos=ins_query_embed)
        ins_hs4 = self.instance_decoder4(ins_tgt, memory, sup_memory=spatial_states_sup, sup_memory_mask=spatial_mask,
                                       memory_key_padding_mask=mask,
                                       pos=pos_embed, query_pos=ins_query_embed)

        ins_hs = (ins_hs4 + ins_hs3 + ins_hs2 + ins_hs1) / 4.0

        ins_hs = ins_hs.transpose(1, 2)

        m_new = torch.mean(memory, dim=0).unsqueeze(dim=1).unsqueeze(dim=0)
        m_new = m_new.repeat(3,1,128,1)
        ins_hs = (ins_hs + m_new) / 2.0


        h_hs = ins_hs[:, :, :num_queries, :]
        o_hs = ins_hs[:, :, num_queries:, :]

        # original
        # ins_guided_embed = (h_hs + o_hs) / 2.0
        # ins_guided_embed = ins_guided_embed.permute(0, 2, 1, 3)
        #
        # inter_tgt = torch.zeros_like(ins_guided_embed[0])
        # inter_hs_ori = self.interaction_decoder(inter_tgt, memory, memory_key_padding_mask=mask,
        #                                         pos=pos_embed, query_pos=ins_guided_embed)
        # inter_hs_ori = inter_hs_ori.transpose(1, 2)
        memory = self.obj_class_ln(self.obj_class_fc(memory))

        # h_hs_detached = h_hs.detach()

        inter_hs = (h_hs + o_hs) / 2.0
        inter_hs = self.queries2spacial_proj(inter_hs[-1])
        inter_hs = self.queries2spacial_proj_norm(inter_hs)
        # inter_hs = inter_hs + self.inter_guided_embedd.weight.unsqueeze(0).repeat(bs, 1, 1)

        dtype = inter_hs.dtype

        clip_cls_feature, clip_visual = clip_model.encode_image(clip_src)
        clip_cls_feature = clip_cls_feature / clip_cls_feature.norm(dim=1, keepdim=True)
        clip_cls_feature = clip_cls_feature.to(dtype)
        clip_visual = clip_visual.to(dtype)
        with torch.no_grad():
            clip_hoi_score = clip_cls_feature @ self.hoi_cls.T
            # obj_score = clip_cls_feature @ self.obj_cls.T
            # obj_hoi_score = obj_score @ self.obj2hoi_proj

            # verb_score = clip_cls_feature @ self.verb_cls.T
            # verb_hoi_score = verb_score @ self.verb2hoi_proj
            # clip_hoi_score += verb_hoi_score * 0.1
            # ignore_idx = clip_hoi_score.sort(descending=True).indices[:, self.topk:]
            # for idx, igx in enumerate(ignore_idx):
            #     clip_hoi_score[idx][igx] *= 0
            clip_hoi_score = clip_hoi_score.unsqueeze(1)

        clip_cls_feature = clip_cls_feature.unsqueeze(1).repeat(1, num_queries, 1)

        inter_hs1 = self.clip_interaction_decoder1(inter_hs.permute(1, 0, 2), clip_visual.permute(1, 0, 2),
                                                  sup_memory=object_states, sup_memory_mask=object_mask, sup_memory2=memory)
        inter_hs2 = self.clip_interaction_decoder2(inter_hs.permute(1, 0, 2), clip_visual.permute(1, 0, 2),
                                                  sup_memory=posture_states, sup_memory_mask=posture_mask, sup_memory2=memory)
        inter_hs3 = self.clip_interaction_decoder3(inter_hs.permute(1, 0, 2), clip_visual.permute(1, 0, 2),
                                                  sup_memory=position_states, sup_memory_mask=position_mask, sup_memory2=memory)
        inter_hs4 = self.clip_interaction_decoder4(inter_hs.permute(1, 0, 2), clip_visual.permute(1, 0, 2),
                                                  sup_memory=spatial_states, sup_memory_mask=spatial_mask, sup_memory2=memory)

        inter_hs = torch.cat((inter_hs1, inter_hs2, inter_hs3, inter_hs4), dim=-1)
        inter_hs = self.inter_class_ln(self.inter_class_fc(inter_hs))

        clip_visual_new = clip_cls_feature.permute(1, 0, 2).unsqueeze(dim=0)
        clip_visual_new = clip_visual_new.repeat(3, 1, 1, 1)
        inter_hs = inter_hs @ clip_proj.to(dtype)
        inter_hs = (inter_hs + clip_visual_new) / 2.0
        inter_hs = inter_hs.permute(0, 2, 1, 3)

        # add
        # ins_guided_embed = (h_hs + o_hs) / 2.0
        # ins_guided_embed = ins_guided_embed.permute(0, 2, 1, 3)
        # #torch.Size([3, 64, 8, 256])
        #
        # inter_tgt = torch.zeros_like(ins_guided_embed[0])
        # inter_hs = self.interaction_decoder(inter_tgt, memory, memory_key_padding_mask=mask,
        #                                     pos=pos_embed, query_pos=ins_guided_embed)
        # inter_hs = inter_hs.transpose(1, 2)

        return h_hs, o_hs, inter_hs, clip_cls_feature, clip_hoi_score, clip_visual @ clip_proj.to(dtype)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderCLIP(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, sup_memory=None, sup_memory_mask=None, sup_memory2=None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for i, layer in enumerate(self.layers):
            if len(output.shape) == 4:
                output = output[i]
            else:
                # only this branch will be used, we only use last human/object query and pass one layer decoder block
                output = output
            output = layer(output, memory, sup_memory=sup_memory, sup_memory_mask=sup_memory_mask, sup_memory2=sup_memory2, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, sup_memory=None, sup_memory_mask=None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for i, layer in enumerate(self.layers):
            if len(query_pos.shape) == 4:
                this_query_pos = query_pos[i]
            else:
                this_query_pos = query_pos
            output = layer(output, memory, sup_memory=sup_memory, sup_memory_mask=sup_memory_mask, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=this_query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, enable_cp=False):
        super().__init__()
        self.enable_cp = enable_cp
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        if self.enable_cp:
            def _inner_forward(args):
                src_inner, q_inner, k_inner, src_mask_inner, src_key_padding_mask_inner = args
                src_inner = self.self_attn(q_inner, k_inner, value=src_inner, attn_mask=src_mask_inner,
                                      key_padding_mask=src_key_padding_mask_inner)[0]
                return src_inner

            src2 = cp.checkpoint(_inner_forward, (src, q, k, src_mask, src_key_padding_mask))
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderFusionLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, enable_cp=False):
        super().__init__()
        self.enable_cp = enable_cp
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.norm4 = LayerNorm(d_model)
        self.norm5 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, sup_memory=None, sup_memory_mask=None, sup_memory2=None,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        if self.enable_cp:
            def _inner_forward(args):
                tgt_inner, q_inner, k_inner, tgt_mask_inner, tgt_key_padding_mask_inner = args
                src_inner = self.self_attn(q_inner, k_inner, value=tgt_inner, attn_mask=tgt_mask_inner,
                                           key_padding_mask=tgt_key_padding_mask_inner)[0]
                return src_inner

            tgt2 = cp.checkpoint(_inner_forward, (tgt, q, k, tgt_mask, tgt_key_padding_mask))
        else:
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if self.enable_cp:
            def _inner_forward_co(args):
                tgt_inner, query_pos_inner, memory_inner, pos_inner, memory_mask_inner, memory_key_padding_mask_inner = args
                src_inner = self.multihead_attn(query=self.with_pos_embed(tgt_inner, query_pos_inner),
                                       key=self.with_pos_embed(memory_inner, pos_inner),
                                       value=memory_inner, attn_mask=memory_mask_inner,
                                       key_padding_mask=memory_key_padding_mask_inner)[0]
                return src_inner

            tgt2 = cp.checkpoint(_inner_forward_co, (tgt, query_pos, memory, pos, memory_mask, memory_key_padding_mask))
        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
        tgt2 = tgt + self.dropout2(tgt2)
        tgt2 = self.norm2(tgt2)

        tgt3 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(sup_memory2, pos),
                                   value=sup_memory2, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        # tgt3 = tgt + self.dropout4(tgt3)
        # tgt3 = self.norm4(tgt3)
        tgt3 = tgt + self.dropout5(tgt3)
        tgt3 = self.norm5(tgt3)

        tgt2 = tgt2 + tgt3

        tgt3 = self.multihead_attn2(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(sup_memory, pos),
                                   value=sup_memory, attn_mask=memory_mask,
                                   key_padding_mask=sup_memory_mask)[0]
        tgt3 = tgt2 + self.dropout4(tgt3)
        tgt = self.norm4(tgt3)
        # tgt3 = tgt + self.dropout2(tgt3)
        # tgt3 = self.norm2(tgt3)

        # tgt = tgt2 + tgt3

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, sup_memory=None, sup_memory_mask=None, sup_memory2=None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, sup_memory, sup_memory_mask, sup_memory2, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, enable_cp=False):
        super().__init__()
        self.enable_cp = enable_cp
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.norm4 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, sup_memory=None, sup_memory_mask=None,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        if self.enable_cp:
            def _inner_forward(args):
                tgt_inner, q_inner, k_inner, tgt_mask_inner, tgt_key_padding_mask_inner = args
                src_inner = self.self_attn(q_inner, k_inner, value=tgt_inner, attn_mask=tgt_mask_inner,
                                           key_padding_mask=tgt_key_padding_mask_inner)[0]
                return src_inner

            tgt2 = cp.checkpoint(_inner_forward, (tgt, q, k, tgt_mask, tgt_key_padding_mask))
        else:
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if self.enable_cp:
            def _inner_forward_co(args):
                tgt_inner, query_pos_inner, memory_inner, pos_inner, memory_mask_inner, memory_key_padding_mask_inner = args
                src_inner = self.multihead_attn(query=self.with_pos_embed(tgt_inner, query_pos_inner),
                                       key=self.with_pos_embed(memory_inner, pos_inner),
                                       value=memory_inner, attn_mask=memory_mask_inner,
                                       key_padding_mask=memory_key_padding_mask_inner)[0]
                return src_inner

            tgt2 = cp.checkpoint(_inner_forward_co, (tgt, query_pos, memory, pos, memory_mask, memory_key_padding_mask))
        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
        tgt2 = tgt + self.dropout2(tgt2)
        tgt2 = self.norm2(tgt2)

        tgt3 = self.multihead_attn2(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(sup_memory, None),
                                   value=sup_memory, attn_mask=memory_mask,
                                   key_padding_mask=sup_memory_mask)[0]
        tgt3 = tgt2 + self.dropout4(tgt3)
        tgt = self.norm4(tgt3)
        # tgt3 = tgt2 + self.dropout2(tgt3)
        # tgt = self.norm2(tgt3)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, sup_memory=None, sup_memory_mask=None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, sup_memory, sup_memory_mask, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_gen(args):
    return GEN(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_dec_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        num_queries=args.num_queries,
        num_inter_dec_layrs=args.inter_dec_layers
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
