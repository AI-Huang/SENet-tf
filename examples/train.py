#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jan-04-21 20:21
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)


import os
import argparse
from datetime import datetime
import tensorflow as tf
from senet.keras_fn.se_resnet import SE_ResNet_50


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs that model has been trained for

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3  # base learning rate
    if 80 <= epoch < 120:
        lr *= 1e-1  # reduce factor
    elif 120 <= epoch < 160:
        lr *= 1e-2
    elif 160 <= epoch < 180:
        lr *= 1e-3
    elif epoch >= 180:
        lr *= 0.5e-3
    print(
        f"Model has been trained for {epoch} epoch(s); learning rate for next epoch: {lr}.")
    tf.summary.scalar('learning rate', data=lr, step=epoch)
    return lr


def get_gpu_memory():
    import subprocess as sp
    def _output_to_list(x): return x.decode('ascii').split('\n')[:-1]

    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0])
                          for i, x in enumerate(memory_free_info)]
    print(memory_free_values)
    return memory_free_values


def get_available_gpu_indices(gpus_memory, required_memory=10240):
    if_enough_memory = [_ > required_memory for _ in gpus_memory]
    available_gpu_indices = [
        i for i, enough in enumerate(if_enough_memory) if enough]
    assert len(available_gpu_indices) >= 2

    return available_gpu_indices


def makedir_exist_ok(dirpath):
    """makedir_exist_ok compatible for both Python 2 and Python 3
    """
    import os
    import six
    import errno

    if six.PY3:
        os.makedirs(
            dirpath, exist_ok=True)  # pylint: disable=unexpected-keyword-arg
    else:
        # Python 2 doesn't have the exist_ok arg, so we try-except here.
        try:
            os.makedirs(dirpath)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def cmd_parser():
    """parse arguments
    """
    parser = argparse.ArgumentParser()

    def string2bool(string):
        """string2bool
        """
        if string not in ["False", "True"]:
            raise argparse.ArgumentTypeError(
                f"""input(={string}) NOT in ["False", "True"]!""")
        if string == "False":
            return False
        elif string == "True":
            return True

    # Input parameters
    parser.add_argument('--dataset_name', type=str, dest='dataset_name',
                        action='store', default="cifar10", help='dataset_name, dataset name, e.g., "cifar10".')

    # Model parameters
    parser.add_argument('--model_type', type=str, dest='model_type',
                        action='store', default="ResNet50", help='model_type, e.g., ResNet50, 1 or 2.')

    # Training parameters
    parser.add_argument('--batch_size', type=int, dest='batch_size',
                        action='store', default=32, help='batch_size, e.g. 32.')  # if using ResNet20v2, 32 for Mac, 32, 64, 128 for server
    parser.add_argument('--epochs', type=int, dest='epochs',
                        action='store', default=150, help='training epochs, e.g. 150.')  # training for 150 epochs is sufficient to fit enough
    parser.add_argument('--if_fast_run', type=string2bool, dest='if_fast_run',
                        action='store', default=False, help='if_fast_run, if True, will only train the model for 3 epochs.')

    # Loss
    parser.add_argument('--loss', type=str, dest='loss',
                        action='store', default="bce", help="""loss name, one of ["bce", "focal"].""")
    parser.add_argument('--start_epoch', type=int, dest='start_epoch',
                        action='store', default=0, help='start_epoch, i.e., epoches that have been trained, e.g. 80.')  # 已经完成的训练数
    parser.add_argument('--ckpt', type=str, dest='ckpt',
                        action='store', default="", help='ckpt, model ckpt file.')

    # Focal loss parameters, only necessary when focal loss is chosen
    parser.add_argument('--alpha', type=float, dest='alpha',
                        action='store', default=0.25, help='alpha pamameter for focal loss if it is used.')
    parser.add_argument('--gamma', type=float, dest='gamma',
                        action='store', default=2, help='gamma pamameter for focal loss if it is used.')

    # Device
    parser.add_argument('--visible_gpu_from', type=int, dest='visible_gpu_from',
                        action='store', default=0, help='visible_gpu_from, the first visible gpu index set by tf.config.')
    parser.add_argument('--model_gpu', type=int, dest='model_gpu',
                        action='store', default=None, help='model_gpu, the number of the model_gpu used for experiment.')
    parser.add_argument('--train_gpu', type=int, dest='train_gpu',
                        action='store', default=None, help='train_gpu, the number of the train_gpu used for experiment.')

    # Other parameters
    parser.add_argument('--tmp', type=string2bool, dest='tmp',
                        action='store', default=False, help='tmp, if true, the yielding data during the training process will be saved into a temporary directory.')
    parser.add_argument('--date_time', type=str, dest='date_time',
                        action='store', default=None, help='date_time, manually set date time, for model data save path configuration.')
    parser.add_argument('--date_time_first', type=string2bool, dest='date_time_first',
                        action='store', default=False, help="date_time_first, if True, make date_time parameter at first in the directories' suffix.")

    args = parser.parse_args()

    return args


def main():
    args = cmd_parser()

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(
        physical_devices[args.visible_gpu_from:], 'GPU')
    gpus_memory = get_gpu_memory()
    available_gpu_indices = get_available_gpu_indices(
        gpus_memory, required_memory=10240)

    # Model type
    model_type = args.model_type
    if model_type not in ["ResNet50", "SE_ResNet_50"]:
        raise ValueError(
            f"""model_type {model_type} not in ["ResNet50", "SE_ResNet_50"].""")
        print("Invalid model type name, quiting...")
        return -1

    model_gpu = available_gpu_indices[0]
    train_gpu = available_gpu_indices[1]

    # Experiment time
    if args.date_time == None:
        date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        date_time = args.date_time

    # Prepare dataset
    dataset_name = args.dataset_name
    if dataset_name == "cifar10":
        data = tf.keras.datasets.cifar10
        num_classes = 10
    elif dataset_name == "cifar100":
        data = tf.keras.datasets.cifar100
        num_classes = 100
    else:
        return

    (train_images, train_labels), (test_images, test_labels) =\
        data.load_data()
    train_labels = tf.keras.utils.to_categorical(train_labels)
    test_labels = tf.keras.utils.to_categorical(test_labels)

    input_shape = train_images.shape[1:]

    # Config paths
    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # prefix = os.path.join(
    # "~", "Documents", "DeepLearningData", dataset_name)
    prefix = "."
    subfix = os.path.join(model_type, date_time)
    ckpt_dir = os.path.expanduser(os.path.join(prefix, "ckpts", subfix))
    log_dir = os.path.expanduser(os.path.join(prefix, "logs", subfix))
    makedir_exist_ok(ckpt_dir)
    makedir_exist_ok(log_dir)

    # Create and compile model
    with tf.device("/device:GPU:" + str(model_gpu)):
        if model_type == "ResNet50":
            model = tf.keras.applications.ResNet50(
                include_top=True,
                weights=None,
                input_shape=input_shape,
                classes=num_classes
            )

        elif model_type == "SE_ResNet_50":
            model = SE_ResNet_50(
                include_top=True,
                weights=None,
                input_shape=input_shape,
                classes=num_classes
            )

    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.metrics import BinaryAccuracy, CategoricalAccuracy
    model.compile(
        loss=tf.losses.CategoricalCrossentropy(),
        optimizer=Adam(lr=lr_schedule(0)),
        metrics=[BinaryAccuracy(), CategoricalAccuracy()]
    )

    # Callbacks
    from tensorflow.keras.callbacks import CSVLogger, TensorBoard
    csv_logger = CSVLogger(os.path.join(
        log_dir, "training.log.csv"), append=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir, histogram_freq=1, update_freq="batch")
    callbacks = [csv_logger, tensorboard_callback]

    # Fit model
    epochs = 3 if args.if_fast_run else args.epochs
    with tf.device("/device:GPU:" + str(train_gpu)):
        model.fit(
            train_images,
            train_labels,
            epochs=epochs,
            validation_data=(test_images, test_labels),
            batch_size=args.batch_size,
            verbose=1, workers=4,
            callbacks=callbacks
        )


if __name__ == "__main__":
    main()
