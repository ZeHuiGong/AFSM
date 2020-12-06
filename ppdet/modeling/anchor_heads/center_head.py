# AUTHOR: Zehui Gong
# DATE: 2020-6-16

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant

from ..backbones.hourglass import _conv_norm, kaiming_init
from .corner_head import mask_feat, nms, _topk
from ppdet.core.workspace import register
import numpy as np
import logging

logger = logging.getLogger(__name__)

__all__ = ['CenterHead', 'CenterHeadIOU']


def ctdet_decode(heat, wh, reg, K=100, batch_size=1):
    """output (Variable): [batch, K, 6] (x1, y1, x2, y2, score, cls)"""
    shape = fluid.layers.shape(heat)
    H, W = shape[2], shape[3]

    heat = fluid.layers.sigmoid(heat)
    heat = nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, batch_size, H, W, K)

    reg = mask_feat(reg, inds, batch_size)
    reg = fluid.layers.reshape(reg, [-1, K, 2])
    xs = fluid.layers.reshape(xs, [-1, K, 1]) + reg[:, :, 0:1]
    ys = fluid.layers.reshape(ys, [-1, K, 1]) + reg[:, :, 1:2]

    wh = mask_feat(wh, inds, batch_size)

    clses = fluid.layers.reshape(clses, [-1, K, 1])
    clses = fluid.layers.cast(clses, 'float32')
    scores = fluid.layers.reshape(scores, [-1, K, 1])
    bboxes = fluid.layers.concat([
        xs - wh[:, :, 0:1] / 2,
        ys - wh[:, :, 1:2] / 2,
        xs + wh[:, :, 0:1] / 2,
        ys + wh[:, :, 1:2] / 2], axis=2)

    return bboxes, scores, clses


def ctdet_decode_iou(heat, wh, reg, K=100, batch_size=1):
    shape = fluid.layers.shape(heat)
    H, W = shape[2], shape[3]

    heat = fluid.layers.sigmoid(heat)
    wh = fluid.layers.exp(wh)
    heat = nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, batch_size, H, W, K)

    reg = mask_feat(reg, inds, batch_size)
    reg = fluid.layers.reshape(reg, [-1, K, 2])
    xs = fluid.layers.reshape(xs, [-1, K, 1]) + reg[:, :, 0:1]
    ys = fluid.layers.reshape(ys, [-1, K, 1]) + reg[:, :, 1:2]

    wh = mask_feat(wh, inds, batch_size)
    bboxes = fluid.layers.concat([
        xs - wh[:, :, 0:1] / 2,
        ys - wh[:, :, 1:2] / 2,
        xs + wh[:, :, 0:1] / 2,
        ys + wh[:, :, 1:2] / 2], axis=2)

    clses = fluid.layers.reshape(clses, [-1, K, 1])
    clses = fluid.layers.cast(clses, 'float32')
    scores = fluid.layers.reshape(scores, [-1, K, 1])

    return bboxes, scores, clses


@register
class CenterHead(object):
    """
    CCenterNet head

    Args:
        train_batch_size(int): batch_size in training process
        test_batch_size(int): batch_size in test process, 1 by default
        num_classes(int): num of classes, 80 by default
        stack(int): stack of hourglass backbone, 2 by default(hourglass-104)
        wh_weight(float): weight of wh regression loss, 0.1 by default
        reg_weight(float): weight of center offset loss, 1 by default
        wh_loss_type(int): the loss class object to compute wh regression penalty.
                      if it is None, the default l1 loss will be used.
        top_k(int): choose top_k centers in prediction, 100 by default
    """
    __shared__ = ['num_classes', 'stack']
    __inject__ = ['wh_loss_type']

    def __init__(self,
                 train_batch_size,
                 test_batch_size=1,
                 num_classes=80,
                 stack=2,
                 wh_weight=0.1,
                 wh_loss_type=None,
                 reg_weight=1,
                 top_k=100):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_classes = num_classes
        self.stack = stack
        self.wh_weight = wh_weight
        self.reg_weight = reg_weight
        self.wh_loss_type = wh_loss_type
        self.K = top_k
        self.heats = []
        self.offs = []
        self.whs = []

    def pred_mod(self, x, dim, name=None):
        conv0 = _conv_norm(
            x, 1, 256, with_bn=False, bn_act='relu', name=name + '_0')
        conv1 = fluid.layers.conv2d(
            input=conv0,
            filter_size=1,
            num_filters=dim,
            param_attr=ParamAttr(
                name=name + "_1_weight", initializer=kaiming_init(conv0, 1)),
            bias_attr=ParamAttr(
                name=name + "_1_bias", initializer=Constant(-2.19)),
            name=name + '_1')
        return conv1

    def get_output(self, input):
        """input: list(Variable)"""
        for ind in range(self.stack):
            cnv = input[ind]

            heat = self.pred_mod(cnv, self.num_classes, name='heats_' + str(ind))
            wh_pred = self.pred_mod(cnv, 2, name='wh_pred_' + str(ind))
            ct_off = self.pred_mod(cnv, 2, name='ct_offs_' + str(ind))

            self.heats.append(heat)
            self.offs.append(ct_off)
            self.whs.append(wh_pred)

    def focal_loss(self, preds, gt):
        preds_clip = []
        # none_pos = fluid.layers.cast(
        #     fluid.layers.reduce_sum(gt_masks) == 0, 'float32')
        # none_pos.stop_gradient = True
        min = fluid.layers.assign(np.array([1e-4], dtype='float32'))
        max = fluid.layers.assign(np.array([1 - 1e-4], dtype='float32'))
        for pred in preds:
            pred_s = fluid.layers.sigmoid(pred)
            pred_min = fluid.layers.elementwise_max(pred_s, min)
            pred_max = fluid.layers.elementwise_min(pred_min, max)
            preds_clip.append(pred_max)

        ones = fluid.layers.ones_like(gt)

        fg_map = fluid.layers.cast(gt == ones, 'float32')
        fg_map.stop_gradient = True
        num_pos = fluid.layers.reduce_sum(fg_map)
        min_num = fluid.layers.ones_like(num_pos)
        num_pos = fluid.layers.elementwise_max(num_pos, min_num)
        num_pos.stop_gradient = True
        bg_map = fluid.layers.cast(gt < ones, 'float32')
        bg_map.stop_gradient = True
        neg_weights = fluid.layers.pow(1 - gt, 4) * bg_map
        neg_weights.stop_gradient = True
        loss = fluid.layers.assign(np.array([0], dtype='float32'))
        for ind, pred in enumerate(preds_clip):
            pos_loss = fluid.layers.log(pred) * fluid.layers.pow(1 - pred,
                                                                 2) * fg_map

            neg_loss = fluid.layers.log(1 - pred) * fluid.layers.pow(
                pred, 2) * neg_weights

            pos_loss = fluid.layers.reduce_sum(pos_loss)
            neg_loss = fluid.layers.reduce_sum(neg_loss)
            focal_loss_ = (neg_loss + pos_loss) / num_pos
            loss -= focal_loss_
        return loss

    def l1_loss(self, pred, target, gt_masks, ind):
        """wh loss, l1_loss version"""
        pred = mask_feat(pred, ind, self.train_batch_size)
        mask = fluid.layers.unsqueeze(gt_masks, [2])
        mask = fluid.layers.expand_as(mask, pred)
        mask = fluid.layers.cast(mask, 'float32')
        mask.stop_gradient = True
        total_num = fluid.layers.reduce_sum(mask)
        total_num.stop_gradient = True
        target.stop_gradient = True
        l1 = fluid.layers.elementwise_sub(pred * mask, target * mask)
        l1 = fluid.layers.abs(l1)
        loss = fluid.layers.reduce_sum(l1)
        return loss / (total_num + 1e-4)

    def get_loss(self, targets):
        gt_heat = targets['heatmaps']
        reg_mask = targets['reg_mask']
        gt_off = targets['regrs']
        gt_ind = targets['ind']
        gt_wh = targets['wh']
        gt_masks = fluid.layers.cast(reg_mask, 'float32')

        # 1. heatmap loss
        focal_loss = 0
        focal_loss_ = self.focal_loss(self.heats, gt_heat)
        focal_loss += focal_loss_

        # 2. loss for wh regression
        wh_loss = 0
        for wh_pred in self.whs:
            if self.wh_loss_type is None:
                wh_loss += self.l1_loss(wh_pred, gt_wh, gt_masks, gt_ind)
            else:
                wh_loss += self.wh_loss_type(
                    wh_pred, gt_wh, gt_masks, gt_ind, self.train_batch_size)
        wh_loss = self.wh_weight * wh_loss

        # 3. loss for offsets regression
        off_loss = 0
        for off_pred in self.offs:
            off_loss += self.l1_loss(off_pred, gt_off, gt_masks, gt_ind)
        off_loss = self.reg_weight * off_loss

        loss = (
                       focal_loss + wh_loss + off_loss) / len(self.heats)
        return {'loss': loss, 'focal_loss': focal_loss,
                'wh_loss': wh_loss, 'off_loss': off_loss}

    def get_prediction(self, input, use_flip=False):
        """Variable"""
        ind = self.stack - 1

        heat = self.pred_mod(input, self.num_classes, name='heats_' + str(ind))
        wh_pred = self.pred_mod(input, 2, name='wh_pred_' + str(ind))
        ct_off = self.pred_mod(input, 2, name='ct_offs_' + str(ind))

        if use_flip:  # flip augmentation
            heat = (heat[0:1] + fluid.layers.flip(heat[1:2], dims=[3])) / 2
            wh_pred = (wh_pred[0:1] + fluid.layers.flip(wh_pred[1:2], dims=[3])) / 2
            ct_off = ct_off[0:1]

        return ctdet_decode(heat, wh_pred, ct_off, K=self.K,
                            batch_size=self.test_batch_size)


@register
class CenterHeadIOU(CenterHead):
    
    def get_prediction(self, input, use_flip=False):
        """Variable"""
        ind = self.stack - 1

        heat = self.pred_mod(input, self.num_classes, name='heats_' + str(ind))
        wh_pred = self.pred_mod(input, 2, name='wh_pred_' + str(ind))
        ct_off = self.pred_mod(input, 2, name='ct_offs_' + str(ind))

        if use_flip:  # flip augmentation
            heat = (heat[0:1] + fluid.layers.flip(heat[1:2], dims=[3])) / 2
            wh_pred = (wh_pred[0:1] + fluid.layers.flip(wh_pred[1:2], dims=[3])) / 2
            ct_off = ct_off[0:1]

        return ctdet_decode_iou(heat, wh_pred, ct_off, K=self.K,
                                batch_size=self.test_batch_size)
