import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Xavier, Constant
from ppdet.experimental import mixed_precision_global_state
import math


class FeatureFusion(object):

    def __init__(self,
                 single_scale=False,
                 spatial_scales=[0.25]):
        super(FeatureFusion, self).__init__()
        self.single_scale = single_scale
        self.spatial_scales = spatial_scales
        if not single_scale:
            self.add_weights = fluid.layers.create_parameter(
                shape=[len(spatial_scales), ],
                dtype='float32',
                default_initializer=fluid.initializer.Constant(1.))
            self.eps = 1e-4

    def __call__(self, inputs):
        """fuse features from different resolutions
               feats (list[Variable])
        """
        # only one features, just upsample
        if self.single_scale:
            fused_feat = inputs[-1]
        else:
            # upsample
            for idx in range(len(inputs) - 1):
                up_scale = int(self.spatial_scales[-1] / self.spatial_scales[idx])
                inputs[idx] = fluid.layers.resize_nearest(inputs[idx], scale=up_scale)
            # normalized weights
            add_weights = fluid.layers.relu(self.add_weights)
            add_weights /= fluid.layers.reduce_sum(add_weights, dim=0, keep_dim=True) + self.eps

            # fuse features across various resolutions
            fused_feat = inputs[0] * add_weights[0]
            for idx in range(1, len(inputs)):
                fused_feat += inputs[idx] * add_weights[idx]

        # final upsample to get features of stride=4
        scale = int(0.25 / self.spatial_scales[-1])
        if scale > 1:
            fused_feat = fluid.layers.resize_nearest(fused_feat, scale=scale)
            fused_feat = fluid.layers.conv2d(
                fused_feat,
                fused_feat.shape[1],
                filter_size=3,
                stride=1,
                padding=1,
                param_attr=ParamAttr(
                    name='centernet_fusefeat_w',
                    initializer=Xavier(fan_out=fused_feat.shape[1])),
                bias_attr=ParamAttr(
                    name='centernet_fusefeat_b',
                    learning_rate=2.),
                name='centernet_fusefeat')
        inputs = [fused_feat]

        return inputs


# below are feature fusion methods, which fuse features across scales in channel dimensions.
class AdaptFeatureFusionV1(object):

    def __init__(self,
                 stride=0.25,
                 spatial_scales=(0.25, ),
                 num_channels=256):
        super(AdaptFeatureFusionV1, self).__init__()
        self.spatial_scales = spatial_scales
        self.num_channels = num_channels
        self.stride = stride
        self.add_weights = fluid.layers.create_parameter(
            shape=[len(spatial_scales), num_channels],
            dtype='float32',
            default_initializer=fluid.initializer.Constant(1.))

    def __call__(self, inputs):
        """fuse features from various resolutions,
        initialize the fusion weights"""
        # upsample
        for idx in range(len(inputs) - 1):
            up_scale = int(self.spatial_scales[-1] / self.spatial_scales[idx])
            inputs[idx] = fluid.layers.resize_nearest(inputs[idx], scale=up_scale)

        # normalized weights
        add_weights = fluid.layers.softmax(self.add_weights, axis=0)

        # fuse features across various resolutions
        fused_feat = inputs[0] * fluid.layers.reshape(add_weights[0], shape=(1, self.num_channels, 1, 1))
        for idx in range(1, len(inputs)):
            fused_feat += inputs[idx] * fluid.layers.reshape(
                add_weights[idx], shape=(1, self.num_channels, 1, 1))

        # final upsample to get features of stride=4
        scale = int(self.stride / self.spatial_scales[-1])
        if scale > 1:
            fused_feat = fluid.layers.resize_nearest(fused_feat, scale=scale)
            fused_feat = fluid.layers.conv2d(
                fused_feat,
                self.num_channels,
                filter_size=3,
                stride=1,
                padding=1,
                param_attr=ParamAttr(
                    name='centernet_fusefeat_w',
                    initializer=Xavier(fan_out=fused_feat.shape[1])),
                bias_attr=ParamAttr(
                    name='centernet_fusefeat_b',
                    learning_rate=2.),
                name='centernet_fusefeat')
        outputs = [fused_feat]
        return outputs


class AdaptFeatureFusionV2(object):

    def __init__(self,
                 stride=0.25,
                 spatial_scales=(0.25,),
                 num_channels=256):
        super(AdaptFeatureFusionV2, self).__init__()
        self.spatial_scales = spatial_scales
        self.num_channels = num_channels
        self.stride = stride

    def squeeze_excitation(self, input, num_channels, reduction_ratio=4, name=None):
        mixed_precision_enabled = mixed_precision_global_state() is not None
        pool = fluid.layers.pool2d(
            input=input,
            pool_size=0,
            pool_type='avg',
            global_pooling=True,
            use_cudnn=mixed_precision_enabled)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        squeeze = fluid.layers.fc(
            input=pool,
            size=int(num_channels / reduction_ratio),
            act='relu',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_sqz_weights'),
            bias_attr=ParamAttr(name=name + '_sqz_offset'))
        stdv = 1.0 / math.sqrt(squeeze.shape[1] * 1.0)
        excitation = fluid.layers.fc(
            input=squeeze,
            size=num_channels,
            act=None,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_exc_weights'),
            bias_attr=ParamAttr(name=name + '_exc_offset'))

        return excitation

    def get_add_weights(self, inputs):
        """use fully connected layers to generate add_weights"""
        add_weights = [self.squeeze_excitation(x, self.num_channels, name='idx{}'.format(idx))
                       for idx, x in enumerate(inputs)]
        add_weights = [fluid.layers.reshape(weight, (1, -1, self.num_channels))
                       for weight in add_weights]
        add_weights = fluid.layers.concat(add_weights, axis=0)
        add_weights = fluid.layers.softmax(add_weights, axis=0)
        return add_weights

    def fuse_features(self, xs, weights):
        fused_feat = xs[0] * fluid.layers.reshape(weights[0], shape=(-1, self.num_channels, 1, 1))
        for idx in range(1, len(xs)):
            fused_feat += xs[idx] * fluid.layers.reshape(
                weights[idx], shape=(-1, self.num_channels, 1, 1))
        return fused_feat

    def __call__(self, inputs):
        """fuse features from various resolutions,
        initialize the fusion weights"""
        # upsample
        for idx in range(len(inputs) - 1):
            up_scale = int(self.spatial_scales[-1] / self.spatial_scales[idx])
            inputs[idx] = fluid.layers.resize_nearest(inputs[idx], scale=up_scale)

        # normalized weights
        add_weights = self.get_add_weights(inputs)

        # fuse features across various resolutions
        fused_feat = self.fuse_features(inputs, weights=add_weights)

        # final upsample to get features of stride=4
        scale = int(self.stride / self.spatial_scales[-1])
        if scale > 1:
            fused_feat = fluid.layers.resize_nearest(fused_feat, scale=scale)
            fused_feat = fluid.layers.conv2d(
                fused_feat,
                self.num_channels,
                filter_size=3,
                stride=1,
                padding=1,
                param_attr=ParamAttr(
                    name='centernet_fusefeat_w',
                    initializer=Xavier(fan_out=fused_feat.shape[1])),
                bias_attr=ParamAttr(
                    name='centernet_fusefeat_b',
                    learning_rate=2.),
                name='centernet_fusefeat')
        outputs = [fused_feat]
        return outputs


class AdaptFeatureFusionV3(AdaptFeatureFusionV2):

    def squeeze_excitation(self, input, num_channels, reduction_ratio=4, name=''):
        stdv = 1.0 / math.sqrt(num_channels * 1.0)
        squeeze = fluid.layers.fc(
            input=input,
            size=int(num_channels / reduction_ratio),
            act='relu',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_sqz_weights'),
            bias_attr=ParamAttr(name=name + '_sqz_offset'))
        stdv = 1.0 / math.sqrt(squeeze.shape[1] * 1.0)
        excitation = fluid.layers.fc(
            input=squeeze,
            size=num_channels,
            act=None,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_exc_weights'),
            bias_attr=ParamAttr(name=name + '_exc_offset'))
        return excitation

    def get_add_weights(self, inputs):
        """use fully connected layers to generate add_weights"""
        # 1. avg_pool
        mixed_precision_enabled = mixed_precision_global_state() is not None
        xs = [fluid.layers.pool2d(input=x, pool_size=0, pool_type='avg', global_pooling=True,
                                  use_cudnn=mixed_precision_enabled)
              for x in inputs]
        xs = fluid.layers.concat(xs, axis=1)
        add_weights = self.squeeze_excitation(xs, self.num_channels * len(inputs))
        add_weights = fluid.layers.reshape(add_weights, (len(inputs), -1, self.num_channels))
        add_weights = fluid.layers.softmax(add_weights, axis=0)
        return add_weights


class AdaptFusionSpitalV1(AdaptFeatureFusionV2):

    def get_add_weights(self, inputs):
        """spatial attention"""
        add_weights = []
        for idx, x in enumerate(inputs):
            weight = fluid.layers.conv2d(
                x, 1, filter_size=1,
                param_attr=ParamAttr(name='spafuse_{}_weights'.format(idx)),
                bias_attr=ParamAttr(name='spafuse_{}_bias'.format(idx)))
            weight = fluid.layers.unsqueeze(weight, axes=[0])
            add_weights.append(weight)
        add_weights = fluid.layers.concat(add_weights, axis=0)
        add_weights = fluid.layers.softmax(add_weights, axis=0)
        return add_weights

    def fuse_features(self, xs, weights):
        fused_feat = xs[0] * weights[0]
        for idx in range(1, len(xs)):
            fused_feat += xs[idx] * weights[idx]
        return fused_feat


class AdaptFusionSpitalV2(AdaptFusionSpitalV1):

    def get_add_weights(self, inputs):
        """spatial attention"""
        add_weights = fluid.layers.concat(inputs, axis=1)
        add_weights = fluid.layers.conv2d(
            add_weights, len(inputs), filter_size=1,
            param_attr=ParamAttr(name='spafuse_weights'),
            bias_attr=ParamAttr(name='spafuse_bias'))
        # (n, 4, h, w) --> (n, 4, 1, h, w)
        add_weights = fluid.layers.unsqueeze(add_weights, axes=[2])
        # (n, 4, 1, h, w) --> (4, n, 1, h, w)
        add_weights = fluid.layers.transpose(add_weights, [1, 0, 2, 3, 4])
        add_weights = fluid.layers.softmax(add_weights, axis=0)
        return add_weights


class AdaptSpaChaFuse(AdaptFeatureFusionV3):

    def get_add_weights(self, inputs):
        # get channel weights
        mixed_precision_enabled = mixed_precision_global_state() is not None
        cha_weights = [fluid.layers.pool2d(input=x, pool_size=0, pool_type='avg', global_pooling=True,
                                  use_cudnn=mixed_precision_enabled)
                       for x in inputs]
        cha_weights = fluid.layers.concat(cha_weights, axis=1)
        cha_weights = self.squeeze_excitation(cha_weights, self.num_channels * len(inputs))
        # (n, 4c) --> (n, 4, c)
        cha_weights = fluid.layers.reshape(cha_weights, (-1, len(inputs), self.num_channels))
        # (n, 4, c) --> (4, n, c)
        cha_weights = fluid.layers.transpose(cha_weights, [1, 0, 2])
        cha_weights = fluid.layers.softmax(cha_weights, axis=0)

        # get spatial weights
        spa_weights = fluid.layers.concat(inputs, axis=1)
        spa_weights = fluid.layers.conv2d(
            spa_weights, len(inputs), filter_size=1,
            param_attr=ParamAttr(name='spafuse_weights'),
            bias_attr=ParamAttr(name='spafuse_bias'))
        # (n, 4, h, w) --> (n, 4, 1, h, w)
        spa_weights = fluid.layers.unsqueeze(spa_weights, axes=[2])
        # (n, 4, 1, h, w) --> (4, n, 1, h, w)
        spa_weights = fluid.layers.transpose(spa_weights, [1, 0, 2, 3, 4])
        spa_weights = fluid.layers.softmax(spa_weights, axis=0)

        return spa_weights, cha_weights

    def fuse_features(self, xs, weights):
        spa_weights, cha_weights = weights

        cha_fused_feat = xs[0] * fluid.layers.reshape(cha_weights[0], shape=(-1, self.num_channels, 1, 1))
        spa_fused_feat = xs[0] * spa_weights[0]
        for idx in range(1, len(xs)):
            cha_fused_feat += xs[idx] * fluid.layers.reshape(
                cha_weights[idx], shape=(-1, self.num_channels, 1, 1))
            spa_fused_feat += xs[idx] * spa_weights[idx]

        fused_feat = spa_fused_feat + cha_fused_feat
        return fused_feat
