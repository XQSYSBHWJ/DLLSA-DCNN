# @Time : 2023/1/5 8:34:38
# @Author : 海晏河清
# @File : 训练网络.py
# The environment here is TensorFlow (tf2)
# conda 4.12.0
# Python 3.7.11
# torch 1.7.1
# tensorflow 2.1.0
#  <(￣︶￣)↗[GO!]
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.preprocessing import StandardScaler as SS
from keras.layers import *
from myattention import *
import keras.layers as KL
import keras
from keras import backend as K
import pandas as pd
import numpy as np
import tensorflow as tf


def cal_pccs(x, y, n):
    """
    warning: data format must be narray
    :param x: Variable 1
    :param y: The variable 2
    :param n: The number of elements in x
    :return: pccs
    """
    sum_xy = np.sum(np.sum(x * y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x * x))
    sum_y2 = np.sum(np.sum(y * y))
    pcc = (n * sum_xy - sum_x * sum_y) / np.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
    return pcc + (1e-6)


def PCC_RMSE(y_true, y_pred):
    alpha = 0.7
    # alpha = args.alpha  # alpha = 0.7
    fsp = y_pred - tf.keras.backend.mean(y_pred)
    fst = y_true - tf.keras.backend.mean(y_true)
    devP = tf.keras.backend.std(y_pred)
    devT = tf.keras.backend.std(y_true)
    rmse = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1))
    pcc = 1.0 - tf.keras.backend.mean(fsp * fst) / (devP * devT)
    loss = alpha * pcc + (1 - alpha) * rmse
    return loss + (1e-6)


def RMSE(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1)) + (1e-6)


def PCC(y_true, y_pred):
    fsp = y_pred - tf.keras.backend.mean(y_pred)
    fst = y_true - tf.keras.backend.mean(y_true)
    devP = tf.keras.backend.std(y_pred)
    devT = tf.keras.backend.std(y_true)
    pcc = tf.keras.backend.mean(fsp * fst) / (devP * devT)
    pcc = tf.where(tf.math.is_nan(pcc), 0.8, pcc)
    return pcc + (1e-6)


def SE_moudle(input_xs, reduction_ratio=16.):
    shape = input_xs.get_shape().as_list()
    se_module = tf.reduce_mean(input_xs, [1, 2])
    # 第一个Dense：shape[-1]/reduction_ratio：即把input_channel再除以reduction_ratio，使channel下降到指定维度数
    se_module = tf.keras.layers.Dense(shape[-1] / reduction_ratio, activation=tf.nn.relu)(se_module)
    # 第二个Dense：重新回升到与input_channel相同的原始维度数
    se_module = tf.keras.layers.Dense(shape[-1], activation=tf.nn.relu)(se_module)
    se_module = tf.nn.sigmoid(se_module)
    se_module = tf.reshape(se_module, [-1, 1, 1, shape[-1]])
    out_ys = tf.multiply(input_xs, se_module)
    return out_ys


# SAM
def spatial_attention(channel_refined_feature):
    maxpool_spatial = KL.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined_feature)
    avgpool_spatial = KL.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined_feature)
    max_avg_pool_spatial = KL.Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    return KL.Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation='sigmoid',
                     kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)


def read_files():
    # 读取文件
    train = pd.read_csv("Onion1_Feature_D2020_all_pka_train.csv", index_col=0)
    valid = pd.read_csv("Onion1_Feature_D2020_all_pka_valid.csv", index_col=0)
    test_1 = pd.read_csv("Onion1_Feature_D2013.csv", index_col=0)
    test_2 = pd.read_csv("Onion1_Feature_D2016.csv", index_col=0)

    X_train = train.values[:, :3840]  # 加载原子残基结构数据
    X_valid = valid.values[:, :3840]
    X_test_1 = test_1.values[:, :3840]
    X_test_2 = test_2.values[:, :3840]

    scaler = SS()    # 标准化
    # 卷积和LSTM网络标准化
    Train_in1 = scaler.fit_transform(X_train).reshape([-1] + [64, 60, 1])
    Valid_in1 = scaler.transform(X_valid).reshape([-1] + [64, 60, 1])
    Test1_in1 = scaler.transform(X_test_1).reshape([-1] + [64, 60, 1])
    Test2_in1 = scaler.transform(X_test_2).reshape([-1] + [64, 60, 1])
    print("普通卷积网络：", Train_in1.shape, Valid_in1.shape, Test1_in1.shape, Test2_in1.shape)
    # 注意力机制标准化
    Train_in2 = scaler.fit_transform(X_train).reshape([-1] + [8, 8, 60])
    Valid_in2 = scaler.transform(X_valid).reshape([-1] + [8, 8, 60])
    Test1_in2 = scaler.transform(X_test_1).reshape([-1] + [8, 8, 60])
    Test2_in2 = scaler.transform(X_test_2).reshape([-1] + [8, 8, 60])

    print('用于spatial_attention的特征标准化:', Train_in2.shape, Valid_in2.shape, Test1_in2.shape, Test2_in2.shape)

    # 对应的真实pka值和标签
    Train_pka = train.pKa.values
    Valid_pka = valid.pKa.values
    Test1_index = test_1.index
    Test2_index = test_2.index

    return Train_in1, Train_in2, Train_pka, \
           Valid_in1, Valid_in2, Valid_pka, \
           Test1_in1, Test1_in2, Test1_index, \
           Test2_in1, Test2_in2, Test2_index

def My_NET():
    # 第一部分，网络结构 ***********************************************************************************  Train_std1
    in_input_1 = Input([64, 60, 1], name='in_input_1')  # in_input_1.shape = (None,62,128,1)
    x1 = Conv2D(32, 4, 1, activation='relu')(in_input_1)
    x1 = Conv2D(64, 4, 1, activation='relu')(x1)
    x1 = Conv2D(128, 4, 1, activation='relu')(x1)

    x1 = TimeDistributed(Flatten())(x1)                  # 双向性能好与单向  x1.shape = (None, 56, 20736)
    x1 = LSTM(32, return_sequences=True)(x1)
    x1 = Activation('relu')(x1)                          # x1.shape = (None, 56, 32)
    x1 = MyAttention(32)(x1)
    x1 = Activation('relu')(x1)                          # x1.shape = (None, 56, 32)
    x1 = BatchNormalization()(x1)

    x1 = Flatten()(x1)

    x1 = Dense(400, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x1)
    x1 = Activation('relu')(x1)
    x1 = Dropout(0.1)(x1)
    x1 = BatchNormalization()(x1)

    x1 = Dense(200, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x1)
    x1 = Activation('relu')(x1)
    x1 = Dropout(0.1)(x1)
    x1 = BatchNormalization()(x1)

    # 第二部分，空间注意力神经网络****************************************************************************** Train_std1
    in_input_2 = Input([8, 8, 60], name='in_input_2')
    x2 = spatial_attention(in_input_2)
    x2 = Conv2D(64, 4, 1, activation='relu')(x2)

    x2 = Flatten()(x2)

    x2 = Dense(200, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x2)
    x2 = Activation('relu')(x2)
    x2 = Dropout(0.1)(x2)
    x2 = BatchNormalization()(x2)

    x2 = Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x2)
    x2 = Activation('relu')(x2)
    x2 = BatchNormalization()(x2)

    # 合并全连接层 ******************************************************************************深度学习

    x_all = concatenate([x1, x2])
    x_all = BatchNormalization()(x_all)

    x_all = Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x_all)
    x_all = Activation('relu')(x_all)
    x_all = BatchNormalization()(x_all)

    x_all = Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x_all)
    output = Activation('relu', name='output')(x_all)

    model = keras.Model(inputs=[in_input_1, in_input_2], outputs=[output])

    # 与上一次不同为lr=0.01
    sgd = tf.keras.optimizers.SGD(lr=0.03, momentum=0.9, decay=1e-6, clipvalue=0.01)
    model.compile(optimizer=sgd, loss={'output': RMSE}, metrics=["mse", PCC, RMSE, PCC_RMSE])
    return model

"""----------------------------------------------------训练模型------------------------------------------------------"""
model = My_NET()
logger = tf.keras.callbacks.CSVLogger("logfile_new.log", separator=',', append=False)
earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=30,
                                                verbose=1, mode='min', restore_best_weights=True)
bestmodel = tf.keras.callbacks.ModelCheckpoint(filepath="bestmodel_0220.h5", verbose=1, save_best_only=True)

DATA = read_files()

# callbacks
history = model.fit(x={'in_input_1': DATA[0],
                       'in_input_2': DATA[1],
                       },
                    y={'output': DATA[2]},
                    epochs=500,
                    batch_size=64,
                    verbose=1,
                    validation_data=([DATA[3], DATA[4]], DATA[5]),
                    callbacks=[earlystopper, bestmodel, logger])
