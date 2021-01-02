#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jan-02-21 20:43
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.platform import test
from tensorflow.keras.utils import plot_model

from senet.keras_fn.se_resnet import SE_ResNet_18, SE_ResNet_50, SE_ResNet_101, SE_ResNet_152


class TestModelArchitectures(keras_parameterized.TestCase):

    def test_se_resnet_18(self):
        model_type = "SE_ResNet_18"

        input_shape = (224, 224, 3)
        num_classes = 2
        model = SE_ResNet_18(
            include_top=True,
            weights=None,
            input_shape=input_shape,
            classes=num_classes
        )

        plot_model(model, to_file=model_type + ".png", show_shapes=True)

    def test_se_resnet_50(self):
        model_type = "SE_ResNet_50"

        input_shape = (224, 224, 3)
        num_classes = 2
        model = SE_ResNet_50(
            include_top=True,
            weights=None,
            input_shape=input_shape,
            classes=num_classes
        )

        plot_model(model, to_file=model_type + ".png", show_shapes=True)

    def test_se_resnet_101(self):
        model_type = "SE_ResNet_101"

        input_shape = (224, 224, 3)
        num_classes = 2
        model = SE_ResNet_101(
            include_top=True,
            weights=None,
            input_shape=input_shape,
            classes=num_classes
        )

        plot_model(model, to_file=model_type + ".png", show_shapes=True)

    def test_se_resnet_152(self):
        model_type = "SE_ResNet_152"

        input_shape = (224, 224, 3)
        num_classes = 2
        model = SE_ResNet_152(
            include_top=True,
            weights=None,
            input_shape=input_shape,
            classes=num_classes
        )

        plot_model(model, to_file=model_type + ".png", show_shapes=True)


if __name__ == "__main__":
    test.main()
