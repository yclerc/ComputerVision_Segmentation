
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.metrics import Recall, Precision
from data import load_data, tf_dataset
import datetime
from model import build_model

#pip install -q -U tensorflow-addons
#import tensorflow_addons as tfa


NAME = "NDT-ULTRASONIMAGES_CEA-{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

def IoU(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    ious=tf.numpy_function(f, [y_true, y_pred], tf.float32)
    return ious


if __name__ == "__main__":
    ## Dataset

    #CEA dataset
    path = "data/"

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)

    ## Hyperparameters
    batch = 8
    lr = 1e-4
    epochs = 20
    #epochs = 50

    train_dataset = tf_dataset(train_x, train_y, batch=batch)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)

    model = build_model()

    opt = tf.keras.optimizers.Adam(lr)
    # Adam: fast, less precise
    # SGD: slow, more precise


    metrics = ["acc", IoU, Recall(), Precision()]
    # TP True Positive, TN True Negative...
    # acc: proportion of correct predictions over total number of predictions (TP+TN)/(TP+TN+FP+FN)
    # IoU: Area of overlap / Area of union
    # Recall: proporion of actual Positive correctly identified(TP)/(TP+FN)
    # Précision: proportion of correctly identified Positive(TP)/(TP+FP)

    # possibility to use IoU as loss function ?
    # exists for bounding boxes, can try to dev custom version


    # mean_iou_loss = tf.Variable(initial_value=-tf.math.log(tf.reduce_sum(IoU())), name='loss', trainable=True)
    # train_op = tf.train.AdamOptimizer(0.001).minimize(mean_iou_loss)

    #model.compile(loss=mean_iou_loss, optimizer="adam", metrics=metrics)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=metrics)


    callbacks = [
        ModelCheckpoint("files/model.h5"),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3),
        CSVLogger("files/data.csv"),
        TensorBoard(log_dir='logs/{}/'.format(NAME+"-"+path)),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=False)
    ]

    # EarlyStopping: Stop training when a monitored metric has stopped improving.

    train_steps = len(train_x)//batch
    valid_steps = len(valid_x)//batch

    if len(train_x) % batch != 0:
        train_steps += 1
    if len(valid_x) % batch != 0:
        valid_steps += 1

    model.fit(train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks)
