#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jan-01-21 17:36
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : https://github.com/hujie-frank/SENet/tree/master/models
# @RefLink : https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/python/keras/applications/resnet.py
# @RefLink : https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py

"""SENet based on tf.keras.applications.resnet
Expanded the foundation codes in TensorFlow **r2.1** .
# Reference:
    - [Squeeze-and-Excitation Networks](
        https://arxiv.org/abs/1709.01507) (CVPR 2018)
# Tested environment:
    tensorflow==2.1.0
    tensorflow==2.2.0
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# for TensorFlow r2.4 using:
# from tensorflow.python.keras.applications.resnet import block1, stack1, ResNet
from keras_applications.resnet_common import ResNet
from keras_applications import get_submodules_from_kwargs
from senet.tf_fn.se_block import se_block
from senet.tf_fn import keras_modules_injection
from tensorflow.python.util.tf_export import keras_export



backend = None
layers = None
models = None
keras_utils = None


@keras_modules_injection
def se_resnet_block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None, **kwargs):
    """A SE-ResNetv1 block.
    Arguments:
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.
    Returns:
        Output tensor for the residual block.
    """
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut:
        shortcut = layers.Conv2D(
            4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(
        filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    # add SE Attention here
    x_attention = se_block(x, name=name + '_se_block')

    x = layers.Add(name=name + '_add')([shortcut, x_attention])
    x = layers.Activation('relu', name=name + '_out')(x)

    return x


@keras_modules_injection
def se_resnet_stack1(x, filters, blocks, stride1=2, name=None, **kwargs):
    """A set of stacked residual blocks.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.
    # Returns
        Output tensor for the stacked blocks.
    """
    x = se_resnet_block1(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = se_resnet_block1(
            x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x


@keras_export('keras.applications.senet.SE_ResNet_18',
              'keras.applications.SE_ResNet_18')
@keras_modules_injection
def SE_ResNet_18(include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1000,
                 **kwargs):
    """Instantiates the SE_ResNet_18 architecture."""

    def stack_fn(x):
        x = se_resnet_stack1(x, 64, 2, stride1=1, name='conv2')
        x = se_resnet_stack1(x, 128, 2, name='conv3')
        x = se_resnet_stack1(x, 256, 2, name='conv4')
        return se_resnet_stack1(x, 512, 2, name='conv5')

    return ResNet(stack_fn, False, True, 'SE_ResNet_18', include_top, weights,
                  input_tensor, input_shape, pooling, classes, **kwargs)


@keras_export('keras.applications.senet.SE_ResNet_50',
              'keras.applications.SE_ResNet_50')
@keras_modules_injection
def SE_ResNet_50(include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1000,
                 **kwargs):
    """Instantiates the SE_ResNet_50 architecture."""

    def stack_fn(x):
        x = se_resnet_stack1(x, 64, 3, stride1=1, name='conv2')
        x = se_resnet_stack1(x, 128, 4, name='conv3')
        x = se_resnet_stack1(x, 256, 6, name='conv4')
        return se_resnet_stack1(x, 512, 3, name='conv5')

    return ResNet(stack_fn, False, True, 'SE_ResNet_50', include_top, weights,
                  input_tensor, input_shape, pooling, classes, **kwargs)


@keras_export('keras.applications.senet.SE_ResNet_101',
              'keras.applications.SE_ResNet_101')
@keras_modules_injection
def SE_ResNet_101(include_top=True,
                  weights='imagenet',
                  input_tensor=None,
                  input_shape=None,
                  pooling=None,
                  classes=1000,
                  **kwargs):
    """Instantiates the SE_ResNet_101 architecture."""

    def stack_fn(x):
        x = se_resnet_stack1(x, 64, 3, stride1=1, name='conv2')
        x = se_resnet_stack1(x, 128, 4, name='conv3')
        x = se_resnet_stack1(x, 256, 23, name='conv4')
        return se_resnet_stack1(x, 512, 3, name='conv5')

    return ResNet(stack_fn, False, True, 'SE_ResNet_101', include_top, weights,
                  input_tensor, input_shape, pooling, classes, **kwargs)


@keras_export('keras.applications.senet.SE_ResNet_152',
              'keras.applications.SE_ResNet_152')
@keras_modules_injection
def SE_ResNet_152(include_top=True,
                  weights='imagenet',
                  input_tensor=None,
                  input_shape=None,
                  pooling=None,
                  classes=1000,
                  **kwargs):
    """Instantiates the SE_ResNet_152 architecture."""

    def stack_fn(x):
        x = se_resnet_stack1(x, 64, 3, stride1=1, name='conv2')
        x = se_resnet_stack1(x, 128, 8, name='conv3')
        x = se_resnet_stack1(x, 256, 36, name='conv4')
        return se_resnet_stack1(x, 512, 3, name='conv5')

    return ResNet(stack_fn, False, True, 'SE_ResNet_152', include_top, weights,
                  input_tensor, input_shape, pooling, classes, **kwargs)
