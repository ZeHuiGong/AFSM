# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

from paddle import fluid

from ppdet.core.workspace import register
import numpy as np
from .AdativeFeatureSelection import FeatureFusion, AdaptFeatureFusionV1
from .cornernet_squeeze import CornerNetSqueeze, rescale_bboxes

__all__ = ['CornerNet']


@register
class CornerNet(CornerNetSqueeze):

    def __init__(self,
                 backbone,
                 corner_head='CornerHead',
                 num_classes=80,
                 fpn=None,
                 single_scale=True,
                 spatial_scales=[0.25]):
        super(CornerNet, self).__init__(
            backbone=backbone,
            corner_head=corner_head,
            num_classes=num_classes,
            fpn=fpn)
        self.single_scale = single_scale
        self.spatial_scales = spatial_scales

    def extract_feat(self, x):
        body_feats = self.backbone(x)
        if self.fpn is not None:
            # the input and output for bifpn are list or tuple
            if self.fpn.__class__.__name__ in ['BiFPN']:
                body_feats = tuple(body_feats.values())
                body_feats = self.fpn(body_feats)
                body_feats = body_feats[::-1]
            else:
                body_feats, _ = self.fpn.get_output(body_feats)
                body_feats = list(body_feats.values())
        else:
            body_feats = list(body_feats.values())

        feature_fusion = FeatureFusion(self.single_scale, self.spatial_scales)
        # feature_fusion = AdaptFeatureFusionV1(spatial_scales=self.spatial_scales,
                                            #   num_channels=body_feats[0].shape[1])
        body_feats = feature_fusion(body_feats)

        return body_feats

    def build(self, feed_vars, mode='train'):
        im = feed_vars['image']
        body_feats = self.extract_feat(im)
        if mode == 'train':
            target_vars = [
                'tl_heatmaps', 'br_heatmaps', 'tag_masks', 'tl_regrs',
                'br_regrs', 'tl_tags', 'br_tags'
            ]
            target = {key: feed_vars[key] for key in target_vars}
            self.corner_head.get_output(body_feats)
            loss = self.corner_head.get_loss(target)
            return loss

        elif mode == 'test':
            ratios = feed_vars['ratios']
            borders = feed_vars['borders']
            bboxes, scores, tl_scores, br_scores, clses = self.corner_head.get_prediction(
                body_feats[-1])
            bboxes = rescale_bboxes(bboxes, ratios, borders)
            detections = fluid.layers.concat([clses, scores, bboxes], axis=2)

            detections = detections[0]
            return {'bbox': detections}

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
            'tl_heatmaps': {'shape': [None, C, output_size[1], output_size[0]], 'dtype': 'float32', 'lod_level': 0},
            'br_heatmaps': {'shape': [None, C, output_size[1], output_size[0]], 'dtype': 'float32', 'lod_level': 0},
            'tl_regrs': {'shape': [None, max_tag_len, 2], 'dtype': 'float32', 'lod_level': 0},
            'br_regrs': {'shape': [None, max_tag_len, 2], 'dtype': 'float32', 'lod_level': 0},
            'tl_tags': {'shape': [None, max_tag_len], 'dtype': 'int64', 'lod_level': 0},
            'br_tags': {'shape': [None, max_tag_len], 'dtype': 'int64', 'lod_level': 0},
            'tag_masks': {'shape': [None, max_tag_len], 'dtype': 'int32', 'lod_level': 0},
            'is_difficult': {'shape': [None, 1],'dtype': 'int32',   'lod_level': 0},
        }
        # yapf: enable
        return inputs_def