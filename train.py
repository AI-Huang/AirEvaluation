#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Aug-24-20 00:42
# @Update  : Oct-07-20 22:58
# @Update  : Dec-04-20 18:40
# @Update  : Dec-27-20 00:23
# @Author  : Kan Huang (kan.huang@connect.ust.hk)


"""Training with WaveNet_LSTM* model
Typical usage:
    python train.py --data_type=pm25 --sensor_index=0 --shuffle=True --stride=10
Environments:
    tensorflow>=2.1.0
"""
import os
import argparse
from datetime import datetime
import tensorflow as tf
# from wavenet.keras_fn.wavenet import
from wavenet.keras_fn.wavenet_lstm import WaveNet_LSTM
from datasets.iaqi_data import load_data
from utils.dir_utils import makedir_exist_ok
from utils.gpu_utils import get_gpu_memory, get_available_gpu_indices


def lr_schedule(epoch):
    lr = 1e-4  # base learning rate
    if epoch >= 20:
        lr *= 0.1  # # reduced by 0.1 when finish training for 40 epochs
    return lr


def training_args():
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

    # Model parameters
    parser.add_argument('--model_type', type=str, dest='model_type',
                        action='store', default="WaveNet_LSTM", choices=["WaveNet_LSTM"], help='tmp, manually set model type, for model data save path configuration.')
    # Model parameters and ablition study
    parser.add_argument('--activation', type=str, dest='activation',
                        action='store', default=None, help="""activation, if not None, it should be activation type such as 'relu'.""")
    parser.add_argument('--batch_norm', type=string2bool, dest='batch_norm',
                        action='store', default="False", help=""".""")
    parser.add_argument('--attention_type', type=str, dest='attention_type',
                        action='store', default="official", choices=["official", "MyAttention", "BahdanauAttention"], help="""attention_type, "custom" or "official".""")

    # Data parameters
    parser.add_argument('--data_type', type=str, dest='data_type',
                        action='store', default="pm25", help='data_type, e.g., --data_type=pm25, data type from the sensor.')
    parser.add_argument('--sensor_index', type=int, dest='sensor_index',
                        action='store', default=0, help='sensor_index, e.g., --sensor_index=0, index of the sensor, 0, 1, 2, 3.')
    parser.add_argument('--all_channels', type=string2bool, dest='all_channels',
                        action='store', default=False, help='all_channels, e.g., --all_channels=False, if True, will train on data from all channels.')
    parser.add_argument('--data_name', type=str, dest='data_name',
                        action='store', default=None, help='data_name, e.g., --data_name=pm25_0, data name from the sensor.')

    parser.add_argument('--window_size', type=int, dest='window_size',
                        action='store', default=7200, help='window_size, e.g., --window_size=7200, window size of the input data.')
    parser.add_argument('--stride', type=int, dest='stride',
                        action='store', default=10, help='stride, e.g., --stride=10, stride of the origin data to yield samples.')
    parser.add_argument('--shuffle', type=string2bool, dest='shuffle',
                        action='store', default=True, help='shuffle, if True, data generator will shuffle after every epoch.')
    parser.add_argument('--seed', type=int, dest='seed',
                        action='store', default=42, help="seed, the initial seed used in WindowSequence's _set_index_array() method.")
    parser.add_argument('--validation_split', type=float,
                        dest='validation_split', action='store', default=0.2,
                        help='validation_split, the split of the validation data.')

    # Training parameters
    parser.add_argument('--batch_size', type=int, dest='batch_size',
                        action='store', default=32, help='batch_size, e.g. 32.')  # --batch_size=32 big enough for 1080 Ti x 4
    parser.add_argument('--epochs', type=int, dest='epochs',
                        action='store', default=100, help='training epochs, e.g. 100.')  # origin exper 100, ETA about 2 hours
    parser.add_argument('--fast_run', type=string2bool, dest='fast_run',
                        action='store', default=False, help='fast_run, if True, will only train the model for 3 epochs.')

    # Device
    parser.add_argument('--visible_gpu_from', type=int, dest='visible_gpu_from',
                        action='store', default=0, help='visible_gpu_from, the first visible gpu index set by tf.config.')
    parser.add_argument('--model_gpu', type=int, dest='model_gpu',
                        action='store', default=None, help='model_gpu, the number of the model_gpu used for experiment.')
    parser.add_argument('--train_gpu', type=int, dest='train_gpu',
                        action='store', default=None, help='train_gpu, the number of the train_gpu used for experiment.')

    # Other parameters
    # Log parameter
    parser.add_argument('--log_prefix', type=str, dest='log_prefix',
                        action='store', default=os.path.expanduser(os.path.join(
                            "~", "Documents", "DeepLearningData", "AirEvaluation")), help='log_prefix, .')
    parser.add_argument('--date_time', type=str, dest='date_time',
                        action='store', default=None, help='date_time, manually set date time, for model data save path configuration.')

    args = parser.parse_args()

    # if data_name is not set
    if not args.data_name:
        args.data_name = "_".join([args.data_type, str(args.sensor_index)])
    pm25_data_list = ["pm25_0", "pm25_1", "pm25_2", "pm25_3", "pm25_all"]
    if not args.data_name in pm25_data_list:
        raise ValueError(f"{args.data_name} NOT in pm25_data_list!")

    return args


def main():
    args = training_args()
    data_name = args.data_name
    window_size = args.window_size
    stride = args.stride
    batch_size = args.batch_size
    validation_split = args.validation_split

    # gpus_memory = get_gpu_memory()
    # available_gpu_indices = get_available_gpu_indices(
    # gpus_memory, required_memory=5000)  # 5000 MB
    # model_gpu = available_gpu_indices[0]
    # train_gpu = available_gpu_indices[1]
    model_gpu, train_gpu = 0, 0
    model_device = "/device:GPU:" + str(model_gpu)
    train_device = "/device:GPU:" + str(train_gpu)

    # else:
    #     model_device = "/device:CPU:0"
    #     train_device = "/device:CPU:0"

    # Load data
    window_sequence_train, window_sequence_val = load_data(
        data_name, window_size, stride, batch_size, validation_split, shuffle=True, seed=42, standard="USA"
    )

    # Config paths
    log_prefix = args.log_prefix
    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    subfix = os.path.join(args.model_type, data_name,
                          "_".join(["stride", str(stride)]), args.attention_type, date_time)  # date_time at last

    # ckpts 和 logs 分开
    log_dir = os.path.join(log_prefix, "logs", subfix)
    ckpt_dir = os.path.join(log_prefix, "ckpts", subfix)
    makedir_exist_ok(log_dir)
    makedir_exist_ok(ckpt_dir)

    # Define callbacks
    from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard

    ckpt_filename = "%s-epoch-{epoch:03d}-mse-{mse:.4f}.h5" % args.model_type
    ckpt_path = os.path.join(ckpt_dir, ckpt_filename)
    checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_path, monitor="mse", verbose=1)

    lr_scheduler = LearningRateScheduler(lr_schedule)
    csv_logger = CSVLogger(os.path.join(
        log_dir, "training.log.csv"), append=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir, histogram_freq=1, update_freq="batch")
    # 不需要 earlystop
    callbacks = [csv_logger, lr_scheduler,
                 checkpoint_callback, tensorboard_callback]

    loss = tf.keras.losses.MeanSquaredError()  # "mse"
    metrics = [  # "mae", "mse", "mape", "msle"
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
        tf.keras.metrics.MeanSquaredError(name="mse"),
        tf.keras.metrics.MeanAbsolutePercentageError(name="mape"),
        tf.keras.metrics.MeanSquaredLogarithmicError(name="msle")
    ]

    # Prepare model
    from tensorflow.keras.optimizers import Adam, SGD
    with tf.device(model_device):
        if args.model_type == "WaveNet_LSTM":
            model = WaveNet_LSTM(input_shape=(
                args.window_size, 1), activation=args.activation, batch_norm=args.batch_norm, attention_type=args.attention_type)

        model.compile(
            Adam(clipvalue=1.0, lr=lr_schedule(0)),
            loss=loss,
            metrics=metrics
        )

    with tf.device(train_device):
        model.fit(
            x=window_sequence_train,
            validation_data=window_sequence_val,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1
        )


if __name__ == "__main__":
    main()
