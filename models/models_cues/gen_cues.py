import json
from PIL import Image
from torch import nn
import torch.nn.functional as F
from pathlib import Path

from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, accuracy, get_world_size,
                       is_dist_avail_and_initialized)
import numpy as np
from ModifiedCLIP import clip
from datasets.hico_text_label import hico_text_label, hico_obj_text_label, hico_unseen_index
from datasets.vcoco_text_label import vcoco_hoi_text_label, vcoco_obj_text_label
from datasets.static_hico import HOI_IDX_TO_ACT_IDX

import torch
import time

from ..backbone import build_backbone
from ..matcher import build_matcher

def _sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y

data_anno = []

class ConCue(nn.Module):
    def __init__(self, backbone, num_queries, aux_loss=False, args=None):
        super().__init__()

        self.args = args
        self.num_queries = num_queries
        hidden_dim = 512
        self.query_embed_h = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_o = nn.Embedding(num_queries, hidden_dim)
        self.pos_guided_embedd = nn.Embedding(num_queries, hidden_dim)
        self.hum_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.inter2verb = MLP(args.clip_embed_dim, args.clip_embed_dim // 2, args.clip_embed_dim, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.dec_layers = self.args.dec_layers

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.obj_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.clip_model, self.preprocess = clip.load(self.args.clip_model)

        if self.args.dataset_file == 'hico':
            hoi_text_label = hico_text_label
            obj_text_label = hico_obj_text_label
            self.obj_text_label = obj_text_label
            unseen_index = hico_unseen_index
        elif self.args.dataset_file == 'vcoco':
            hoi_text_label = vcoco_hoi_text_label
            obj_text_label = vcoco_obj_text_label
            self.obj_text_label = obj_text_label
            unseen_index = None

        clip_label, obj_clip_label, v_linear_proj_weight, hoi_text, obj_text, train_clip_label = \
            self.init_classifier_with_CLIP(hoi_text_label, obj_text_label, unseen_index, args.no_clip_cls_init)

        if self.args.del_unseen and unseen_index is not None:
            self.hoi_text_label_del = []
            unseen_index_list = unseen_index.get(self.args.zero_shot_type, [])
            for idx, k in enumerate(hoi_text_label.keys()):
                if idx in unseen_index_list:
                    continue
                else:
                    self.hoi_text_label_del.append(hoi_text_label[k])
        else:
            self.hoi_text_label_del = [hoi_text_label[k] for idx, k in enumerate(hoi_text_label.keys())]
        num_obj_classes = len(obj_text) - 1  # del nothing
        self.clip_visual_proj = v_linear_proj_weight

        self.hoi_class_fc = nn.Sequential(
            nn.Linear(hidden_dim, args.clip_embed_dim),
            nn.LayerNorm(args.clip_embed_dim),
        )

        if unseen_index:
            unseen_index_list = unseen_index.get(self.args.zero_shot_type, [])
        else:
            unseen_index_list = []

        if self.args.dataset_file == 'hico':
            verb2hoi_proj = torch.zeros(117, 600)
            select_idx = list(set([i for i in range(600)]) - set(unseen_index_list))
            for idx, v in enumerate(HOI_IDX_TO_ACT_IDX):
                verb2hoi_proj[v][idx] = 1.0
            self.verb2hoi_proj = nn.Parameter(verb2hoi_proj[:, select_idx], requires_grad=False)
            self.verb2hoi_proj_eval = nn.Parameter(verb2hoi_proj, requires_grad=False)

            self.verb_projection_model = nn.Linear(args.clip_embed_dim, 117, bias=False)
        else:
            verb2hoi_proj = torch.zeros(29, 263)
            for i in vcoco_hoi_text_label.keys():
                verb2hoi_proj[i[0]][i[1]] = 1

            self.verb2hoi_proj = nn.Parameter(verb2hoi_proj, requires_grad=False)
            self.verb_projection_model = nn.Linear(args.clip_embed_dim, 29, bias=False)

        if args.with_obj_clip_label:
            self.obj_class_fc = nn.Sequential(
                nn.Linear(hidden_dim, args.clip_embed_dim),
                nn.LayerNorm(args.clip_embed_dim),
            )
            if args.fix_clip_label:
                self.obj_visual_projection = nn.Linear(args.clip_embed_dim, num_obj_classes + 1, bias=False)
                self.obj_visual_projection.weight.data = obj_clip_label / obj_clip_label.norm(dim=-1, keepdim=True)
                for i in self.obj_visual_projection.parameters():
                    i.require_grads = False
            else:
                self.obj_visual_projection = nn.Linear(args.clip_embed_dim, num_obj_classes + 1)
                self.obj_visual_projection.weight.data = obj_clip_label / obj_clip_label.norm(dim=-1, keepdim=True)
        else:
            self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)

        self.hidden_dim = hidden_dim
        self.reset_parameters()

        if args.with_clip_label:
            self.visual_projection_model = nn.Linear(args.clip_embed_dim, len(hoi_text))
            for i in self.visual_projection_model.parameters():
                i.require_grads = False
            self.visual_projection_model.weight.data = clip_label / clip_label.norm(dim=-1, keepdim=True)

            if self.args.dataset_file == 'hico' and self.args.zero_shot_type != 'default':
                self.eval_visual_projection_model = nn.Linear(args.clip_embed_dim, 600, bias=False)
                for i in self.eval_visual_projection_model.parameters():
                    i.require_grads = False
                self.eval_visual_projection_model.weight.data = clip_label / clip_label.norm(dim=-1, keepdim=True)
        else:
            self.hoi_class_embedding_model = nn.Linear(args.clip_embed_dim, len(hoi_text))

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.inter2verb = MLP(args.clip_embed_dim, args.clip_embed_dim // 2, args.clip_embed_dim, 3)

    def reset_parameters(self):
        nn.init.uniform_(self.pos_guided_embedd.weight)

    def init_classifier_with_CLIP(self, hoi_text_label, obj_text_label, unseen_index, no_clip_cls_init=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        text_inputs = torch.cat([clip.tokenize(hoi_text_label[id]) for id in hoi_text_label.keys()])
        # print(hoi_text_label.keys())
        if self.args.del_unseen and unseen_index is not None:
            hoi_text_label_del = {}
            unseen_index_list = unseen_index.get(self.args.zero_shot_type, [])
            for idx, k in enumerate(hoi_text_label.keys()):
                if idx in unseen_index_list:
                    continue
                else:
                    hoi_text_label_del[k] = hoi_text_label[k]
        else:
            hoi_text_label_del = hoi_text_label.copy()
        text_inputs_del = torch.cat(
            [clip.tokenize(hoi_text_label[id]) for id in hoi_text_label_del.keys()])

        obj_text_inputs = torch.cat([clip.tokenize(obj_text[1]) for obj_text in obj_text_label])
        clip_model = self.clip_model
        clip_model.to(device)
        with torch.no_grad():
            text_embedding = clip_model.encode_text(text_inputs.to(device))
            text_embedding_del = clip_model.encode_text(text_inputs_del.to(device))
            obj_text_embedding = clip_model.encode_text(obj_text_inputs.to(device))
            v_linear_proj_weight = clip_model.visual.proj.detach()

        if not no_clip_cls_init:
            print('\nuse clip text encoder to init classifier weight\n')
            return text_embedding.float(), obj_text_embedding.float(), v_linear_proj_weight.float(), \
                   hoi_text_label_del, obj_text_inputs, text_embedding_del.float()
        else:
            print('\nnot use clip text encoder to init classifier weight\n')
            return torch.randn_like(text_embedding.float()), torch.randn_like(
                obj_text_embedding.float()), torch.randn_like(v_linear_proj_weight.float()), \
                   hoi_text_label_del, obj_text_inputs, torch.randn_like(text_embedding_del.float())

    def init_classifier(self, hoi_text_label, no_clip_cls_init=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # print(hoi_text_label)
        text_inputs = clip.tokenize(hoi_text_label)

        clip_model = self.clip_model
        clip_model.to(device)
        with torch.no_grad():
            text_embedding = clip_model.encode_text(text_inputs.to(device))
        return text_embedding.float()

    def forward(self, samples: NestedTensor, tokenizer, llama_model, is_training=True, clip_input=None, targets=None,
                args=None):
        device = torch.device("cuda:0")
        root = Path(args.hoi_path)

        if args.visual_cues == 'posture':
            prompt = """
# Your task is to analyze an image that may feature multiple individuals and objects. Focus on the body language of these individuals, describing their poses and facial expressions to understand their states.
# Please limit your description to two sentences.
# Answer:
# """
        elif args.visual_cues == 'blip':
            prompt = """
# Please analyze the image below and identify the interactions between person(s) and object(s) observed within the image.
# <Output Format>
# Each response should be in the format: The person is [interaction label] [object label]
# <Output Example>
# The person is playing a sports ball.
# The person is holding a laptop.
# """
        else:
            print("select a prompt.")

        for index in range(len(targets)):
            if not args.eval:
                if args.dataset_file == 'hico':
                    image_path = root / 'images' / 'train2015' / targets[index]['filename']
                else:
                    image_path = root / 'images' / 'train2014' / targets[index]['filename']
            else:
                if args.dataset_file == 'hico':
                    image_path = root / 'images' / 'test2015' / targets[index]['filename']
                else:
                    image_path = root / 'images' / 'val2014' / targets[index]['filename']
            image = Image.open(image_path)

            start_time = time.time()
            inputs = tokenizer(images=image, text=prompt, return_tensors="pt").to(device)

            generated_ids = llama_model.generate(
                **inputs,
                max_length=256,
                temperature=0.6,
            )
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            data_dict = {}
            data_dict['filename'] = targets[index]['filename']
            data_dict['output'] = generated_text
            data_anno.append(data_dict)
            end_time = time.time()
            print('generate llm cost %f ' % (end_time - start_time))
        seen_inter_dumps = json.dumps(data_anno)
        if args.eval:
            file_name = args.visual_cues + '_test.json'
            save_path = root / 'annotations' / file_name
        else:
            file_name = args.visual_cues + '.json'
            save_path = root / 'annotations' / file_name
        a = open(save_path, 'w')
        a.write(seen_inter_dumps)
        a.close()
        return data_anno

    def _neg_loss(self, pred, gt, weights=None, alpha=0.25):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        if weights is not None:
            pos_loss = pos_loss * weights[:-1]

        neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    @torch.jit.unused
    def _set_aux_loss_triplet(self, outputs_obj_class,
                              outputs_sub_coord, outputs_obj_coord, outputs_inter_hs=None):

        # if outputs_hoi_class_model.shape[0] == 1:
        #     outputs_hoi_class_model = outputs_hoi_class_model.repeat(self.dec_layers, 1, 1, 1)
        aux_outputs = {'pred_obj_logits': outputs_obj_class[-self.dec_layers: -1],
                       'pred_sub_boxes': outputs_sub_coord[-self.dec_layers: -1],
                       'pred_obj_boxes': outputs_obj_coord[-self.dec_layers: -1],
                       }
        if outputs_inter_hs is not None:
            aux_outputs['inter_memory'] = outputs_inter_hs[-self.dec_layers: -1]
        outputs_auxes = []
        for i in range(self.dec_layers - 1):
            output_aux = {}
            for aux_key in aux_outputs.keys():
                output_aux[aux_key] = aux_outputs[aux_key][i]
            outputs_auxes.append(output_aux)

        return outputs_auxes

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


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


class SetCriterionHOI(nn.Module):

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses, args):
        super().__init__()

        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.with_mimic:
            self.clip_model, _ = clip.load(args.clip_model, device=device)
        else:
            self.clip_model = None
        self.alpha = args.alpha

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_obj_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_interactions):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.sigmoid()
        loss_verb_ce = self._neg_loss(src_logits, target_classes, weights=None, alpha=self.alpha)
        losses = {'loss_verb_ce': loss_verb_ce}
        return losses

    def loss_hoi_labels_model(self, outputs, targets, indices, num_interactions, topk=5):
        assert 'pred_hoi_logits_model' in outputs
        src_logits = outputs['pred_hoi_logits_model']
        dtype = src_logits.dtype

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['hoi_labels'][J] for t, (_, J) in zip(targets, indices)]).to(dtype)
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o
        src_logits = _sigmoid(src_logits)
        loss_hoi_ce = self._neg_loss(src_logits, target_classes, weights=None, alpha=self.alpha)
        losses = {'loss_hoi_labels_model': loss_hoi_ce}

        _, pred = src_logits[idx].topk(topk, 1, True, True)
        acc = 0.0
        for tid, target in enumerate(target_classes_o):
            tgt_idx = torch.where(target == 1)[0]
            if len(tgt_idx) == 0:
                continue
            acc_pred = 0.0
            for tgt_rel in tgt_idx:
                acc_pred += (tgt_rel in pred[tid])
            acc += acc_pred / len(tgt_idx)
        rel_labels_error = 100 - 100 * acc / max(len(target_classes_o), 1)
        losses['hoi_class_error_model'] = torch.from_numpy(np.array(
            rel_labels_error)).to(src_logits.device).float()
        return losses

    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (
                    exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses

    def mimic_loss(self, outputs, targets, indices, num_interactions):
        src_feats = outputs['inter_memory']
        src_feats = torch.mean(src_feats, dim=1)

        target_clip_inputs = torch.cat([t['clip_inputs'].unsqueeze(0) for t in targets])
        with torch.no_grad():
            target_clip_feats = self.clip_model.encode_image(target_clip_inputs)[0]
        loss_feat_mimic = F.l1_loss(src_feats, target_clip_feats)
        losses = {'loss_feat_mimic': loss_feat_mimic}
        return losses

    def reconstruction_loss(self, outputs, targets, indices, num_interactions):
        raw_feature = outputs['clip_cls_feature']
        hoi_feature = outputs['hoi_feature']

        loss_rec = F.l1_loss(raw_feature, hoi_feature)
        return {'loss_rec': loss_rec}

    def _neg_loss(self, pred, gt, weights=None, alpha=0.25):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        if weights is not None:
            pos_loss = pos_loss * weights[:-1]

        neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        loss_map = {
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        num_interactions = sum(len(t['hoi_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float,
                                           device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions))

        return losses


class PostProcessHOITriplet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.subject_category_id = args.subject_category_id

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_hoi_logits_model = outputs['pred_hoi_logits_model']
        out_hoi_logits_llm = outputs['pred_hoi_logits_llm']
        out_obj_logits = outputs['pred_obj_logits']
        out_sub_boxes = outputs['pred_sub_boxes']
        out_obj_boxes = outputs['pred_obj_boxes']
        clip_visual = outputs['clip_visual']
        # clip_logits = outputs['clip_logits']

        assert len(out_hoi_logits_model) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        hoi_scores_model = out_hoi_logits_model.sigmoid()
        hoi_scores_llm = out_hoi_logits_llm.sigmoid()
        hoi_scores = hoi_scores_model + hoi_scores_llm
        obj_scores = out_obj_logits.sigmoid()
        obj_labels = F.softmax(out_obj_logits, -1)[..., :-1].max(-1)[1]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(hoi_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes).to(hoi_scores.device)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes).to(hoi_scores.device)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for index in range(len(hoi_scores)):
            hs, os, ol, sb, ob = hoi_scores[index], obj_scores[index], obj_labels[index], sub_boxes[index], obj_boxes[
                index]
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            ids = torch.arange(b.shape[0])

            results[-1].update(
                {'hoi_scores': hs.to('cpu'), 'obj_scores': os.to('cpu'), 'clip_visual': clip_visual[index].to('cpu'),
                 'sub_ids': ids[:ids.shape[0] // 2], 'obj_ids': ids[ids.shape[0] // 2:]})

        return results



def build(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)

    model = ConCue(
        backbone,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        args=args
    )

    llama_model = InstructBlipForConditionalGeneration.from_pretrained(
        "./data/instructblip-vicuna-7b/").cuda()

    tokenizer = InstructBlipProcessor.from_pretrained("./data/instructblip-vicuna-7b/")

    matcher = build_matcher(args)
    weight_dict = {}
    if args.with_clip_label:
        weight_dict['loss_hoi_labels_model'] = args.hoi_loss_coef
        weight_dict['loss_obj_ce'] = args.obj_loss_coef
    else:
        weight_dict['loss_hoi_labels_model'] = args.hoi_loss_coef
        weight_dict['loss_obj_ce'] = args.obj_loss_coef

    weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
    weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
    weight_dict['loss_sub_giou'] = args.giou_loss_coef
    weight_dict['loss_obj_giou'] = args.giou_loss_coef
    if args.with_mimic:
        weight_dict['loss_feat_mimic'] = args.mimic_loss_coef

    if args.with_rec_loss:
        weight_dict['loss_rec'] = args.rec_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.with_mimic:
        losses = ['feats_mimic']

    losses = ['feats_mimic']

    if args.with_rec_loss:
        losses.append('rec_loss')

    criterion = SetCriterionHOI(args.num_obj_classes, args.num_queries, args.num_verb_classes, matcher=matcher,
                                weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses,
                                args=args)
    criterion.to(device)
    postprocessors = {'hoi': PostProcessHOITriplet(args)}

    return model, criterion, postprocessors, tokenizer, llama_model
