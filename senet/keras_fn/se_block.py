#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Dec-01-20 21:54
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : https://github.com/hujie-frank/SENet/tree/master/models

"""SE Block based on tf.keras.applications.resnet
# Reference:
    - [Squeeze-and-Excitation Networks](
        https://arxiv.org/abs/1709.01507) (CVPR 2018)
# Tested environment:
    tensorflow==2.1.0
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense


def se_block(input_):
    """A SENet block implementation with Keras functional API
    Input:
        input_: 2D feature maps with shape (H, W, C)
    """
    num_channels = input_.shape[-1]
    x = GlobalAveragePooling2D()(input_)
    x = tf.expand_dims(x, -2)
    x = tf.expand_dims(x, -2)  # make output shape be (None, 1, 1, C)
    x = Dense(num_channels, activation="relu")(x)
    # The last FC layer generates the scale (or query) tensor
    x = Dense(num_channels, activation="sigmoid")(x)

    return x * input_  # multiply (None, 1, 1, C) and (None, H, W, C)
