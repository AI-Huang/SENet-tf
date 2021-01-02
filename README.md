# SENet

TensorFlow implementation for SENet, attached with some experiments.

## Installation and usage

Clone this repository

```bash
git clone https://github.com/AI-Huang/SENet
```

and install

```bash
cd SENet
python setup.py install
```

The module is installed calling `senet` as in `setup.py`. Using example:

```Python
from senet.keras_fn.se_resnet import SE_ResNet_18
```

## SENet family

| model           | based on | in original paper |
| --------------- | -------- | ----------------- |
| SE-BN-Inception | -------- | Y                 |
| SE-ResNet-18    | ResNet   | N                 |
| SE-ResNet-50    | ResNet   | Y                 |
| SE-ResNet-101   | ResNet   | N                 |
| SE-ResNet-152   | ResNet   | Y                 |
| SE-ResNeXt-50   | ResNeXt  | Y                 |
| SE-ResNeXt-101  | ResNeXt  | Y                 |
| SENet-154       | -------- | Y                 |

## References

[1]. [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)

[2]. [https://github.com/hujie-frank/SENet](https://github.com/hujie-frank/SENet)
