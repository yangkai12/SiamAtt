# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss, atts_loss, offs_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck
from pysot.models.atts import att_mods_head
from pysot.models.offset import off_mods_head
class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build attention map
        self.att_mods_head = att_mods_head(cfg.atts.TYPE,
                                 **cfg.atts.KWARGS)

        # build offs map
        self.off_mods_head = off_mods_head(cfg.offs.TYPE,
                                **cfg.offs.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        # build mask head
        if cfg.MASK.MASK:
            self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                           **cfg.MASK.KWARGS)

            if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)

    def template(self, z):
        zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        atts, attentions = self.att_mods_head_(self.zf, xf) # 对应3个不同深度的网络输出 atts:目标中心点位置 attentions：注意力特征图
        #atts = self.off_mods_head_(self.zf, xf, atts)
        cls, loc = self.rpn_head(self.zf, xf) # cls:[1,10,25,25] loc:[1,20,25,25]

        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
                'atts': atts,
                'attentions': attentions,
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None
               }
    '''
    def off_mods_head_(self, zf, xf, atts):
        offs = self.off_mods_head(zf, xf) # [1,2,25,25]*3
        i, j = 0, 0
        for off, att in zip(offs, atts):
            for att_ in att:
                atts[i][j][0] = att_[0].astype(int) + off[
                    0, 0, att_[0].astype(int), att_[1].astype(int)].cpu().detach().numpy() # x
                atts[i][j][1] = att_[1].astype(int) + off[
                    0, 1, att_[0].astype(int), att_[1].astype(int)].cpu().detach().numpy() # y
                j += 1
            i += 1
            j = 0
        return atts
    '''

    def att_mods_head_(self, zf, xf):
        attentions = self.att_mods_head(zf, xf) # [1,1,25,25]*3
        atts = [torch.sigmoid(att) for att in attentions]
        att_nms_ks = cfg.atts.att_nms_ks
        atts_score = self.att_nms(atts, att_nms_ks)  # atts 进行非最大值抑制
        atts = self.decode_atts(atts_score) # 满足threshold的位置
        return atts, attentions

    def decode_atts(self, atts):
        thresh = cfg.atts.thresh
        att_ = []
        for att in atts:
            att_1 = att.cpu()
            ys, xs = np.where(att_1[0, 0] > thresh)  # process locations where scores are above a threshold 0.3
            scores_att = att_1[0, 0, ys, xs]

            att_.append(np.stack((ys, xs, scores_att.detach()), axis=1))

        return att_

    def att_nms(self, atts, ks):
        pads = [(k - 1) // 2 for k in ks]
        pools = [nn.functional.max_pool2d(att, (k, k), stride=1, padding=pad) for k, att, pad in
                 zip(ks, atts, pads)]  # 通过池化层将图片缩小
        keeps = [(att == pool).float() for att, pool in zip(atts, pools)]
        atts = [att * keep for att, keep in zip(atts, keeps)]
        return atts

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()
        label_atts = data['atts'].cuda()
        label_tag_masks = data['tag_masks'].cuda()
        label_tl_regrs = data['tl_regrs'].cuda()
        label_tl_tags = data['tl_tags'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        # attention
        atts = self.att_mods_head(zf, xf)  # 不同层网络特征得到的attention map [32,1,25,25]*3
        offs = self.off_mods_head(zf, xf) # offs:[28,2,25,25]
        cls, loc = self.rpn_head(zf, xf) # zf:template xf:search

        # get loss
        att_loss = atts_loss(atts, label_atts)
        off_loss = offs_loss(offs, label_tag_masks, label_tl_regrs, label_tl_tags)
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss + att_loss + off_loss  # 总的loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['att_loss'] = att_loss
        outputs['off_loss'] = off_loss

        if cfg.MASK.MASK:
            # TODO
            mask, self.mask_corr_feature = self.mask_head(zf, xf)
            mask_loss = None
            outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
            outputs['mask_loss'] = mask_loss
        return outputs
