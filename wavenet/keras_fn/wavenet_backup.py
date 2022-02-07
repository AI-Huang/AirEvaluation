#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Dec-04-20 17:07
# @Author  : Kan Huang (kan.huang@connect.ust.hk)

"""WaveNet implemented with Keras functional API
Environments:
tensorflow>=2.1.0
"""
import tensorflow as tf
from tensorflow.keras import Input, initializers, regularizers, constraints
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, LSTM, Conv1D, Multiply, Add, AveragePooling1D, Bidirectional, Dropout, Dense, Attention, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from wavenet.keras_fn.attention_layers import MyAttention, BahdanauAttention


# 正确版本


def wave_block(x, filters, kernel_size, n, batch_norm=False):
    """WaveNet Residual Conv1D block
    Inputs:
        x:
        filters:
        kernel_size:
        n:
        batch_norm:
    Return:
    """
    # Dilated Conv
    dilation_rates = [2**i for i in range(n)]
    for dilation_rate in dilation_rates:
        residual_x = x
        tanh_out = Conv1D(filters=filters,
                          kernel_size=kernel_size,
                          padding='same',
                          activation='tanh',
                          dilation_rate=dilation_rate,
                          kernel_regularizer=l2(1e-4))(x)
        sigm_out = Conv1D(filters=filters,
                          kernel_size=kernel_size,
                          padding='same',
                          activation='sigmoid',
                          dilation_rate=dilation_rate,
                          kernel_regularizer=l2(1e-4))(x)
        x = Multiply()([tanh_out, sigm_out])
        x = Conv1D(1, 1)(x)  # Skip connections
        if batch_norm:
            x = BatchNormalization(beta_regularizer=l2(
                1e-4), gamma_regularizer=l2(1e-4))(x)
        x = Add()([residual_x, x])

    return x  # Not residual_x!

# 错误版本


def wave_block_ver2020(x, filters, kernel_size, n, batch_norm=False):
    """WaveNet Residual Conv1D block
    Inputs:
        x:
        filters:
        kernel_size:
        n:
        batch_norm:
    Return:
    """
    # Causal Conv
    x = Conv1D(filters=filters,  # should be 1
               kernel_size=1,
               padding='same',
               kernel_regularizer=l2(1e-4))(x)

    # Dilated Conv
    dilation_rates = [2**i for i in range(n)]
    for dilation_rate in dilation_rates:
        residual_x = x  # Stored as residual node
        tanh_out = Conv1D(filters=filters,
                          kernel_size=kernel_size,
                          padding='same',
                          activation='tanh',
                          dilation_rate=dilation_rate,
                          kernel_regularizer=l2(1e-4))(x)
        sigm_out = Conv1D(filters=filters,
                          kernel_size=kernel_size,
                          padding='same',
                          activation='sigmoid',
                          dilation_rate=dilation_rate,
                          kernel_regularizer=l2(1e-4))(x)
        x = Multiply()([tanh_out, sigm_out])
        x = Conv1D(1, 1)(x)  # Skip connections
        if batch_norm:
            x = BatchNormalization(beta_regularizer=l2(
                1e-4), gamma_regularizer=l2(1e-4))(x)
        x = Add()([residual_x, x])

    return residual_x  # Should have returned x


def WaveNet_LSTM_ver2020(input_shape, activation=None, batch_norm=False, attention_type="MyAttention"):
    """WaveNet_LSTM
    Inputs
        input_shape: the input must be a vector, so *input_shape* must be (dim, 1).
    """

    # hparams, n=8 means the receptive field is 2**(8+1)=512
    base_filters = 16
    # Each layer's number of filters is twice as the former layer
    layer_filters = [base_filters, base_filters*2, base_filters*4]
    kernel_size = 3
    dilation_layers = [8, 5, 3]

    input_ = Input(shape=input_shape)

    x = wave_block_ver2020(input_, layer_filters[0], kernel_size,
                           dilation_layers[0], batch_norm=batch_norm)
    if activation:
        x = Activation(activation)(x)
    x = AveragePooling1D(10)(x)
    if batch_norm:
        x = BatchNormalization(beta_regularizer=l2(
            1e-4), gamma_regularizer=l2(1e-4))(x)

    x = wave_block_ver2020(x, layer_filters[1], kernel_size,
                           dilation_layers[1], batch_norm=batch_norm)
    if activation:
        x = Activation(activation)(x)
    x = AveragePooling1D(10)(x)
    if batch_norm:
        x = BatchNormalization(beta_regularizer=l2(
            1e-4), gamma_regularizer=l2(1e-4))(x)

    x = wave_block_ver2020(x, layer_filters[2], kernel_size,
                           dilation_layers[2], batch_norm=batch_norm)
    if activation:
        x = Activation(activation)(x)
    x = AveragePooling1D(10)(x)
    if batch_norm:
        x = BatchNormalization(beta_regularizer=l2(
            1e-4), gamma_regularizer=l2(1e-4))(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)

    valid_attention = ["official", "MyAttention", "BahdanauAttention"]
    assert attention_type in valid_attention
    if attention_type == valid_attention[0]:
        x = Attention()([x, x])
    elif attention_type == valid_attention[1]:
        x = MyAttention(input_shape[0]//1000)(x)
    elif attention_type == valid_attention[2]:
        x = BahdanauAttention(input_shape[0]//1000)(x)

    x = Dropout(0.2)(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(1)(x)
    model = Model(inputs=input_, outputs=x)

    return model


def main():
    from tensorflow.keras.utils import plot_model

    model = WaveNet_LSTM_ver2020(input_shape=(7200, 1),
                                 activation=None, attention_type="MyAttention")
    plot_model(model, to_file=f"./png/WaveNet_LSTM-46.png", show_shapes=True)


if __name__ == "__main__":
    main()
