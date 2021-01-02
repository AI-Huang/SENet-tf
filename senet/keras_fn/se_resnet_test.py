#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jan-02-21 20:43
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import tensorflow as tf
from senet.keras_fn.se_resnet import SENet20


def main():
    model_type = "SENet20"
    input_shape = (224, 224, 3)
    num_classes = 2
    model = SENet20(
        include_top=True,
        weights=None,
        input_shape=input_shape,
        classes=num_classes
    )

    from tensorflow.keras.utils import plot_model
    plot_model(model, to_file=model_type + ".png", show_shapes=True)


if __name__ == "__main__":
    main()
