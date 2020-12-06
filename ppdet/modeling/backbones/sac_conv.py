from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from paddle import fluid

from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype
from paddle.fluid.dygraph.layer_object_helper import LayerObjectHelper

from paddle.fluid.initializer import Normal, Constant, NumpyArrayInitializer

from paddle.fluid.param_attr import ParamAttr, WeightNormParamAttr

from paddle.fluid.framework import Variable
from paddle.fluid.layers import utils


def Fconv2d(input,
            filter,
            stride=1,
            padding=0,
            dilation=1,
            groups=None,
            use_cudnn=True,
            bias_attr=None,
            name=None):
    """
    Similar with conv2d, this is a convolution2D layers. Difference
    is filter can be token as input directly instead of setting filter size
    and number of fliters. Filter is a  4-D tensor with shape
    [num_filter, num_channel, filter_size_h, filter_size_w].
   """
    helper = LayerHelper("conv2d_with_filter", **locals())
    num_channels = input.shape[1]
    num_filters = filter.shape[0]
    num_filter_channels = filter.shape[1]
    l_type = 'conv2d'
    # if (num_channels == groups and
    if (num_channels == groups and num_filters % num_channels == 0 and
            not use_cudnn):
        l_type = 'depthwise_conv2d'
    if groups is None:
        assert num_filter_channels == num_channels
        groups = 1
    else:
        if num_channels % groups != 0:
            raise ValueError("num_channels must be divisible by groups.")
        if num_channels // groups != num_filter_channels:
            raise ValueError("num_filter_channels must equal to num_channels\
                              divided by groups.")

    stride = utils.convert_to_list(stride, 2, 'stride')
    padding = utils.convert_to_list(padding, 2, 'padding')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')
    if not isinstance(use_cudnn, bool):
        raise ValueError("use_cudnn should be True or False")
    pre_bias = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type=l_type,
        inputs={
            'Input': input,
            'Filter': filter,
        },
        outputs={"Output": pre_bias},
        attrs={
            'strides': stride,
            'paddings': padding,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'use_mkldnn': False
        })

    output = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)
    return output


def conv_offset(input,
                filter_size,
                stride,
                padding,
                act=None,
                name=None):
    out_channel = filter_size * filter_size * 3
    offset_mask = fluid.layers.conv2d(
        input,
        num_filters=out_channel,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        param_attr=ParamAttr(
            initializer=Constant(0.0), name=name + ".w_0"),
        bias_attr=ParamAttr(
            initializer=Constant(0.0), name=name + ".b_0"),
        act=act)
    offset_channel = filter_size ** 2 * 2
    mask_channel = filter_size ** 2
    offset, mask = fluid.layers.split(
        input=offset_mask,
        num_or_sections=[offset_channel, mask_channel],
        dim=1)
    mask = fluid.layers.sigmoid(mask)
    return offset, mask


def Fdeformable_conv(input,
                     offset,
                     mask,
                     weight,
                     stride=1,
                     padding=0,
                     dilation=1,
                     groups=None,
                     deformable_groups=None,
                     im2col_step=None,
                     modulated=True,
                     bias_attr=None,
                     name=None):
    """
    Similar with defromable_conv, this is a deformable convolution2D layers. Difference
    is filter can be token as input directly instead of setting filter size
    and number of fliters. Filter is a  4-D tensor with shape
    [num_filter, num_channel, filter_size_h, filter_size_w].
    """
    check_variable_and_dtype(input, "input", ['float32', 'float64'],
                             'deformable_conv')
    check_variable_and_dtype(offset, "offset", ['float32', 'float64'],
                             'deformable_conv')
    check_type(mask, 'mask', (Variable, type(None)), 'deformable_conv')

    num_channels = input.shape[1]
    helper = LayerHelper('deformable_conv', **locals())
    dtype = helper.input_dtype()

    num_filters = weight.shape[0]
    num_filter_channels = weight.shape[1]

    if not isinstance(input, Variable):
        raise TypeError("Input of deformable_conv must be Variable")
    if not isinstance(offset, Variable):
        raise TypeError("Input Offset of deformable_conv must be Variable")

    if groups is None:
        assert num_filter_channels == num_channels
        num_filter_channels = num_channels
        groups = 1
    else:
        if num_channels % groups != 0:
            raise ValueError("num_channels must be divisible by groups.")
        num_filter_channels = num_channels // groups

    stride = utils.convert_to_list(stride, 2, 'stride')
    padding = utils.convert_to_list(padding, 2, 'padding')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')

    pre_bias = helper.create_variable_for_type_inference(dtype)

    if modulated:
        helper.append_op(
            type='deformable_conv',
            inputs={
                'Input': input,
                'Filter': weight,
                'Offset': offset,
                'Mask': mask,
            },
            outputs={"Output": pre_bias},
            attrs={
                'strides': stride,
                'paddings': padding,
                'dilations': dilation,
                'groups': groups,
                'deformable_groups': deformable_groups,
                'im2col_step': im2col_step,
            })

    else:
        helper.append_op(
            type='deformable_conv_v1',
            inputs={
                'Input': input,
                'Filter': weight,
                'Offset': offset,
            },
            outputs={"Output": pre_bias},
            attrs={
                'strides': stride,
                'paddings': padding,
                'dilations': dilation,
                'groups': groups,
                'deformable_groups': deformable_groups,
                'im2col_step': im2col_step,
            })
    output = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)
    return output


def get_shape(input):
    shape = fluid.layers.shape(input)
    shape_hw = fluid.layers.slice(shape, axes=[0], starts=[2], ends=[4])
    out_shape_ = shape_hw
    out_shape = fluid.layers.cast(out_shape_, dtype='int32')
    out_shape.stop_gradient = True
    return out_shape


def SAConv2d(input,
             num_filters,
             filter_size,
             stride=1,
             padding=0,
             dilation=1,
             groups=None,
             bias_attr=False,
             use_cudnn=True,
             act=None,
             name=None,
             lr_mult=1.,
             use_deform=False):
    """switchable convolution"""
    num_channels = input.shape[1]
    if groups is None:
        num_filter_channels = num_channels
    else:
        if num_channels % groups != 0:
            raise ValueError(
                "the channel of input must be divisible by groups,"
                "received: the channel of input is {}, the shape of input is {}"
                ", the groups is {}".format(num_channels, input.shape, groups))
        num_filter_channels = num_channels // groups

    dtype = input.dtype
    filter_size = utils.convert_to_list(filter_size, 2, 'filter_size')
    filter_shape = [num_filters, int(num_filter_channels)] + filter_size

    def _get_default_param_initializer():
        filter_elem_num = filter_size[0] * filter_size[1] * num_channels
        std = (2.0 / filter_elem_num)**0.5
        return Normal(0.0, std, 0)

    weight = fluid.layers.create_parameter(
        attr=ParamAttr(name=name + '_weights',
                       learning_rate=lr_mult),
        shape=filter_shape,
        dtype=dtype,
        default_initializer=_get_default_param_initializer())
    weight_diff = fluid.layers.create_parameter(
        attr=ParamAttr(initializer=Constant(0.),
                       name=name + '_weights_diff',
                       learning_rate=lr_mult),
        shape=filter_shape,
        dtype=dtype,
        default_initializer=_get_default_param_initializer())

    # pre-context
    avg_input = fluid.layers.adaptive_pool2d(
        input=input,
        pool_size=1,
        pool_type='avg')
    avg_input = fluid.layers.conv2d(
        avg_input,
        num_filters=num_channels,
        filter_size=1,
        param_attr=ParamAttr(
            name=name + '_preContext.w',
            learning_rate=lr_mult,
            initializer=Constant(0.)),
        bias_attr=ParamAttr(
            name=name + '_preContext.b',
            learning_rate=lr_mult,
            initializer=Constant(0)),
        act=None)
    shape_hw = get_shape(input)
    avg_input = fluid.layers.expand(avg_input, [1, 1, shape_hw[0], shape_hw[1]])
    input = fluid.layers.elementwise_add(input, avg_input)

    # switch
    avg_input = fluid.layers.pad2d(input, paddings=[2, 2, 2, 2], mode='reflect')
    avg_input = fluid.layers.pool2d(avg_input, pool_size=5, pool_type='avg',
                                    pool_stride=1, pool_padding=0)
    switch = fluid.layers.conv2d(
        avg_input,
        num_filters=1,
        filter_size=1,
        stride=stride,
        param_attr=ParamAttr(
            name=name + '_switch.w',
            learning_rate=lr_mult,
            initializer=Constant(0.)),
        bias_attr=ParamAttr(
            name=name + '_switch.b',
            learning_rate=lr_mult,
            initializer=Constant(1.)),
        act='sigmoid')

    # sac conv
    if use_deform:
        offset_s, mask_s = conv_offset(
            avg_input, 3, stride=stride, padding=1, name=name + '_s_offset')
        out_s = Fdeformable_conv(
            input, offset_s, mask_s, weight,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, deformable_groups=1, im2col_step=1,
            bias_attr=False, name=name + '_s_.conv2d.output.1')
    else:
        out_s = Fconv2d(input, weight, stride, padding, dilation, groups,
                        bias_attr=False, name=name + '_s_.conv2d.output.1')
    padding *= 3
    dilation *= 3

    weight_l = fluid.layers.elementwise_add(weight, weight_diff)
    if use_deform:
        offset_l, mask_l = conv_offset(
            avg_input, 3, stride=stride, padding=1, name=name + '_l_offset')
        out_l = Fdeformable_conv(
            input, offset_l, mask_l, weight_l,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, deformable_groups=1, im2col_step=1,
            bias_attr=False, name=name + '_l_.conv2d.output.1')
    else:
        out_l = Fconv2d(input, weight_l, stride, padding, dilation, groups,
                        bias_attr=False, name=name + '_l_.conv2d.output.1')
    out = switch * out_s + (1 - switch) * out_l

    # post-context
    avg_out = fluid.layers.adaptive_pool2d(out, pool_size=1, pool_type='avg')
    avg_out = fluid.layers.conv2d(
        avg_out, num_filters, filter_size=1,
        param_attr=ParamAttr(initializer=Constant(0.),
                             name=name + '_postContext.w',
                             learning_rate=lr_mult),
        bias_attr=ParamAttr(initializer=Constant(0.),
                            name=name + '_postContext.b',
                            learning_rate=lr_mult))
    shape_hw = get_shape(out)
    avg_out = fluid.layers.expand(avg_out, [1, 1, shape_hw[0], shape_hw[1]])
    out = fluid.layers.elementwise_add(out, avg_out)
    return out
