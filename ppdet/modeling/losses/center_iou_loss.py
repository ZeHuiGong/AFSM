# Author: Zehui Gong
# Date: 2020/6/27

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from paddle import fluid
from ppdet.core.workspace import register, serializable
from ppdet.modeling.anchor_heads.corner_head import mask_feat

__all__ = ['CenterIOULoss']


@register
@serializable
class CenterIOULoss(object):
    """
       Using iou loss for bounding box regression
       Args:
           with_ciou_term (bool): whether to add complete iou term (considering the w/h ratio)
           iou_loss_type: wether to use giou to replace the normal iou
    """

    def __init__(self,
                 iou_loss_type='iou',
                 with_ciou_term=False):
        assert iou_loss_type in ['iou', 'linear_iou', 'giou'], \
            "expected 'iou | giou | linear_iou', but got {}".format(iou_loss_type)
        self.iou_loss_type = iou_loss_type
        self.with_ciou_term = with_ciou_term

    def _transpose_reshape(self, pred, target, gt_mask):
        # (n, max_len, c) --> (-1, c)
        # pred = fluid.layers.transpose(pred, [0, 2, 3, 1])
        # target = fluid.layers.transpose(target, [0, 2, 3, 1])
        # gt_mask = fluid.layers.transpose(gt_mask, [0, 2, 3, 1])

        pred = fluid.layers.reshape(pred, [-1, 2])
        target = fluid.layers.reshape(target, [-1, 2])
        gt_mask = fluid.layers.reshape(gt_mask, [-1, 2])

        return pred, target, gt_mask

    def __call__(self, pred, target, gt_mask, ind, bs):
        """
            Args:
                pred (Variable) : predicted wh [n, 2, H, W]
                target (Variable): ground-truth wh  [n, max_len, 2] (w h)
                gt_mask (Variable): [n, max_len] a mask represents
                the foreground points that need to compute loss during training
        """
        eps = 1e-4
        pred = mask_feat(pred, ind, bs)
        pred = fluid.layers.exp(pred)

        mask = fluid.layers.cast(gt_mask, 'float32')
        avg_factor = fluid.layers.reduce_sum(mask)
        avg_factor.stop_gradient = True

        mask = fluid.layers.unsqueeze(mask, [2])
        mask = fluid.layers.expand_as(mask, pred)

        pred, target, mask = self._transpose_reshape(pred, target, mask)
        mask.stop_gradient = True
        target.stop_gradient = True

        pred = pred * mask
        target = target * mask
        inter_wh = fluid.layers.elementwise_min(pred, target)

        inter_area = inter_wh[:, 0] * inter_wh[:, 1]
        pred_area = pred[:, 0] * pred[:, 1]
        tar_area = target[:, 0] * target[:, 1]

        ious = (inter_area + eps) / (pred_area + tar_area - inter_area + eps)
        if self.iou_loss_type.lower() == 'linear_iou':
            loss = 1.0 - ious
        elif self.iou_loss_type.lower() == 'giou':
            enclose_wh = fluid.layers.elementwise_max(pred, target)
            enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]
            area_union = pred_area + tar_area - inter_area
            gious = ious - (enclose_area - area_union + eps) / (enclose_area + eps)
            loss = 1.0 - gious
        else:
            loss = 0.0 - fluid.layers.log(ious + eps)

        loss = fluid.layers.reduce_sum(loss * mask[:, 0])
        return loss / (avg_factor + eps)
