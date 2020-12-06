# AUTHOR: Zehui Gong
# DATE: 2020/6/21

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Xavier, Constant

from ppdet.core.workspace import register
from ppdet.utils.check import check_version
from .centernet import CenterNet


__all__ = ['RFPCenterNet']


class ASPP(object):

    def __init__(self, out_channels):
        super(ASPP, self).__init__()
        self.out_channels = out_channels
        self.kernel_sizes = [1, 3, 3, 1]
        self.dilations = [1, 3, 6, 1]
        self.paddinds = [0, 3, 6, 0]
        self.aspp_num = len(self.kernel_sizes)

    def __call__(self, x, name=''):
        avg_x = fluid.layers.adaptive_pool2d(x, 1, pool_type="avg", name=name + 'aspp_ada_gap')
        outs = []
        for idx in range(self.aspp_num):
            inp = avg_x if (idx == self.aspp_num - 1) else x
            out = fluid.layers.conv2d(
                inp,
                self.out_channels,
                filter_size=self.kernel_sizes[idx],
                stride=1,
                padding=self.paddinds[idx],
                dilation=self.dilations[idx],
                param_attr=ParamAttr(name=name + 'aspp_conv{}.w'.format(idx)),
                bias_attr=ParamAttr(initializer=Constant(0),
                                    name=name + 'aspp_conv{}.b'.format(idx)),
                act='relu',
                name=name + 'aspp_conv{}'.format(idx))
            outs.append(out)
        outs[-1] = fluid.layers.expand(outs[-1], [1, 1, outs[0].shape[2], outs[0].shape[3]])
        out = fluid.layers.concat(outs, axis=1)

        return out


@register
class RFPCenterNet(CenterNet):
    """Recursive feature pyramid. details in https://arxiv.org/abs/2006.02334.
    Args:
            single_scale (bool): a flag that represents whether use single scale feature (e.g., level3)
            or multi-scale feature fusion (fuse features across various resolutions) to predict
            the final heatmap and size.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head='CenterHead',
                 rfp_steps=2,
                 fpn_levels=(),
                 stage_with_rfp=[3, 4, 5],
                 rfp_sharing=False,
                 num_classes=80,
                 single_scale=True):
        check_version('1.8.0')
        assert neck is not None, 'in recursive feature pyramid, you must have a fpn neck!'
        super(RFPCenterNet, self).__init__(
            backbone,
            neck,
            head,
            num_classes,
            single_scale)

        fpnidx_list = list(range(fpn_levels[1], fpn_levels[0]-1, -1))
        self.stage2fpnidx = {stage: fpnidx_list.index(stage) for stage in stage_with_rfp}
        self.rfp_steps = rfp_steps
        self.rfp_sharing = rfp_sharing
        self.stage_with_rfp = stage_with_rfp
        self.rfp_aspp = ASPP(neck.num_chan // 4)

    def extract_feat(self, x):
        input_w = x.shape[-1]
        # step 1
        self.backbone.prefix_name = ''
        body_feats = self.backbone(x)
        body_feats, _ = self.neck.get_output(body_feats)
        body_feats = tuple(body_feats.values())

        # feedback connection (recursive refinement)
        for rfp_idx in range(self.rfp_steps - 1):
            rfp_feats = tuple(self.rfp_aspp(body_feats[self.stage2fpnidx[i]])
                              for i in range(2, 6) if i in self.stage_with_rfp)
            if self.rfp_sharing:  # sharing backbone parameters
                body_feats_idx = self.backbone(x, rfp_feats)
            else:
                self.backbone.prefix_name = 'rfp_step{}'.format(rfp_idx + 1)
                body_feats_idx = self.backbone(x, rfp_feats)
            body_feats_idx, _ = self.neck.get_output(body_feats_idx)
            body_feats_idx = tuple(body_feats_idx.values())
            body_feats_new = []
            # feature fusion, fuse features from
            for ft_idx in range(len(body_feats_idx)):
                add_weight = self.rfp_weight(body_feats_idx[ft_idx])
                body_feats_new.append(add_weight * body_feats_idx[ft_idx] +
                                      (1-add_weight) * body_feats[ft_idx])
            body_feats = body_feats_new
        body_feats = self.fuse_features(body_feats, input_w)

        return body_feats

    def rfp_weight(self, feat, name=''):
        add_weight = fluid.layers.conv2d(
            feat,
            1,
            filter_size=1,
            stride=1,
            padding=0,
            param_attr=ParamAttr(
                initializer=Constant(0),
                name=name + 'rfp_weight.w'),
            bias_attr=ParamAttr(
                initializer=Constant(0),
                name=name + 'rfp_weight.b'),
            name=name + 'rfp_weight')
        add_weight = fluid.layers.sigmoid(add_weight)

        return add_weight
