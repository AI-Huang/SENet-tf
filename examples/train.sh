#!/bin/sh
# @Date    : Jan-04-21 20:33
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

date_time=$(date "+%Y%m%d-%H%M%S")

# python ./examples/train.py --model_type=ResNet50 --dataset_name=cifar10 --batch_size=32 --epochs=150 --date_time=${date_time}
python ./examples/train.py --model_type=SE_ResNet_50 --dataset_name=cifar10 --batch_size=32 --epochs=150 --date_time=${date_time}
