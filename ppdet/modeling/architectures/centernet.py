# AUTHOR: Zehui Gong
# DATE: 2020/6/16

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import copy

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Xavier, Constant

from ppdet.core.workspace import register
import numpy as np
from ppdet.utils.check import check_version
from .cornernet_squeeze import rescale_bboxes
from .input_helper import corner_multiscale_def
from .AdativeFeatureSelection import FeatureFusion, AdaptFeatureFusionV1

__all__ = ['CenterNet']


@register
class CenterNet(object):
    """Args:
            single_scale (bool): a flag that represents whether use single scale feature (e.g., level3)
            or multi-scale feature fusion (fuse features across various resolutions) to predict
            the final heatmap and size.
    """
    __category__ = 'architecture'
    __inject__ = ['backbone', 'neck', 'head']
    __shared__ = ['num_classes']

    def __init__(self,
                 backbone,
                 neck=None,
                 head='CenterHead',
                 num_classes=80,
                 single_scale=True,
                 spatial_scales=[0.25]):
        check_version('1.8.0')
        super(CenterNet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.num_classes = num_classes
        self.single_scale = single_scale
        self.spatial_scales = spatial_scales

    def extract_feat(self, x):
        body_feats = self.backbone(x)
        if self.neck is not None:
            # the input and output for bifpn are list or tuple
            if self.neck.__class__.__name__ in ['BiFPN']:
                body_feats = tuple(body_feats.values())
                body_feats = self.neck(body_feats)
                body_feats = body_feats[::-1]
            else:
                body_feats, _ = self.neck.get_output(body_feats)
                body_feats = list(body_feats.values())
        else:
            body_feats = list(body_feats.values())

        # feature_fusion = FeatureFusion(self.single_scale, self.spatial_scales)
        feature_fusion = AdaptFeatureFusionV1(spatial_scales=self.spatial_scales, 
                                              num_channels=body_feats[0].shape[1])
        body_feats = feature_fusion(body_feats)

        return body_feats

    def build(self, feed_vars, mode='train'):
        im = feed_vars['image']
        body_feats = self.extract_feat(im)
        if mode == 'train':
            target_vars = ['heatmaps', 'reg_mask', 'ind', 'wh', 'regrs']  # heat_weight
            target = {key: feed_vars[key] for key in target_vars}
            self.head.get_output(body_feats)
            loss = self.head.get_loss(target)
            return loss

        elif mode == 'test':
            ratios = feed_vars['ratios']
            borders = feed_vars['borders']
            bboxes, scores, clses = self.head.get_prediction(body_feats[-1])
            bboxes = rescale_bboxes(bboxes, ratios, borders)
            detections = fluid.layers.concat([clses, scores, bboxes], axis=2)
            detections = detections[0]
            return {'bbox': detections}

    def build_multi_scale(self, feed_vars):
        results = {}
        for i, scale in enumerate(self.test_scales):
            im_name = 'image_scale_{}'.format(scale)
            ratio_name = 'ratios_' + im_name
            border_name = 'borders_' + im_name
            # sizes_name = 'sizes_' + im_name
            img = feed_vars[im_name]
            ratios = feed_vars[ratio_name]
            borders = feed_vars[border_name]
            # sizes = feed_vars[sizes_name]
            if self.use_flip:
                im_name_flip = 'image_flip_scale_{}'.format(scale)
                im_flip = feed_vars[im_name_flip]
                img = fluid.layers.concat([img, im_flip], axis=0)
            body_feats = self.extract_feat(img)
            bboxes, scores, clses = self.head.get_prediction(
                body_feats[-1], use_flip=self.use_flip)
            bboxes = rescale_bboxes(bboxes, ratios, borders)
            bboxes = bboxes / scale
            detection = fluid.layers.concat([clses, scores, bboxes], axis=2)
            det_name = 'bbox_scale_{}'.format(scale) 
            results[det_name] = detection[0]
        return results

    def _input_check(self, require_fields, feed_vars):
        for var in require_fields:
            assert var in feed_vars, \
                "{} has no {} field".format(feed_vars, var)
                
    def _inputs_def(self, image_shape, output_size, max_tag_len):
        """output_size: (w, h)"""
        im_shape = [None] + image_shape
        C = self.num_classes
        # yapf: disable
        inputs_def = {
            'image': {'shape': im_shape, 'dtype': 'float32', 'lod_level': 0},
            'im_id': {'shape': [None, 1], 'dtype': 'int64', 'lod_level': 0},
            'gt_bbox': {'shape': [None, 4], 'dtype': 'float32', 'lod_level': 1},
            'gt_class': {'shape': [None, 1], 'dtype': 'int32', 'lod_level': 1},
            'ratios': {'shape': [None, 2], 'dtype': 'float32', 'lod_level': 0},
            'borders': {'shape': [None, 4], 'dtype': 'float32', 'lod_level': 0},
            'sizes': {'shape': [None, 2], 'dtype': 'float32', 'lod_level': 0},
            'heatmaps': {'shape': [None, C, output_size[1], output_size[0]], 'dtype': 'float32', 'lod_level': 0},
            'regrs': {'shape': [None, max_tag_len, 2], 'dtype': 'float32', 'lod_level': 0},
            'reg_mask': {'shape': [None, max_tag_len], 'dtype': 'float32', 'lod_level': 0},
            'ind': {'shape': [None, max_tag_len], 'dtype': 'int64', 'lod_level': 0},
            'wh': {'shape': [None, max_tag_len, 2], 'dtype': 'float32', 'lod_level': 0},
            'tlbr': {'shape': [None, 2, output_size[1], output_size[0]], 'dtype': 'float32', 'lod_level': 0},
            'tlbr_mask': {'shape': [None, 1, output_size[1], output_size[0]], 'dtype': 'float32', 'lod_level': 0},
            'heat_weight': {'shape': [None, C, output_size[1], output_size[0]], 'dtype': 'float32', 'lod_level': 0},
            'is_difficult': {'shape': [None, 1], 'dtype': 'int32', 'lod_level': 0},
        }
        # yapf: enable
        return inputs_def

    def build_inputs(
            self,
            image_shape=[3, None, None],
            fields=[
                'image', 'im_id', 'gt_box', 'gt_class', 'heatmaps',
                'regrs', 'reg_mask', 'ind', 'wh',
            ],  # for train
            multi_scale=False,
            test_scales=[1.0],
            use_flip=None,
            output_size=[128, 128],
            max_tag_len=128,
            use_dataloader=True,
            iterable=False):
        inputs_def = self._inputs_def(image_shape, output_size, max_tag_len)
        fields = copy.deepcopy(fields)
        if multi_scale:
            ms_def, ms_fields = corner_multiscale_def(image_shape, test_scales, use_flip)
            inputs_def.update(ms_def)
            fields += ms_fields
            self.use_flip = use_flip
            self.test_scales = test_scales

        feed_vars = OrderedDict([(key, fluid.data(
            name=key,
            shape=inputs_def[key]['shape'],
            dtype=inputs_def[key]['dtype'],
            lod_level=inputs_def[key]['lod_level'])) for key in fields])
        loader = fluid.io.DataLoader.from_generator(
            feed_list=list(feed_vars.values()),
            capacity=64,
            use_double_buffer=True,
            iterable=iterable) if use_dataloader else None
        return feed_vars, loader

    def train(self, feed_vars):
        return self.build(feed_vars, mode='train')

    def eval(self, feed_vars, multi_scale=None):
        if multi_scale:
            return self.build_multi_scale(feed_vars)
        return self.build(feed_vars, mode='test')

    def test(self, feed_vars):
        return self.build(feed_vars, mode='test')
