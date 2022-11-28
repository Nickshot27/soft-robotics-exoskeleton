import pdb
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import math as mt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import keras
import keras_tuner as kt

tf.random.set_seed(42)

data=pd.read_csv('Datos.csv')
x=data.iloc[:,[0,1,2,3,4,5,6,7]].values #promedios
y=data.iloc[:,8].values #canal activado

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=5)

x_train = x_train/1669.0
x_test = x_test/1669.0

modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(356, input_shape= (8,), activation="relu"), #42, 500, 356
    tf.keras.layers.Dense(740, activation="relu"), #38, 100, 908
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation="softmax") #5
])

modelo.compile(  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                 metrics=[tf.metrics.SparseCategoricalAccuracy()])

history = modelo.fit(x_train,
                       y_train,
                       epochs=200, #101
                       validation_data=(x_test, y_test))
val_acc_per_epoch = history.history['val_sparse_categorical_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))