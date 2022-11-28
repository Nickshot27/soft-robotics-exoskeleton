import keras_tuner as kt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import math as mt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import keras
tf.random.set_seed(42)

data=pd.read_csv('Datos.csv')
x=data.iloc[:,[0,1,2,3,4,5,6,7]].values #promedios
y=data.iloc[:,8].values #canal activado

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state= 5)

x_train = x_train/1669.0
x_test = x_test/1669.0


def model_builder(hp):
  model = keras.Sequential()

  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-512
  model.add(keras.layers.Dense(units=356, activation='relu'))
  hp_units = hp.Int('units', min_value=8, max_value=1200, step=4)
  model.add(keras.layers.Dense(units=hp_units, activation='relu'))
  model.add(keras.layers.Dropout(0.2))
  model.add(keras.layers.Dense(units=5, activation='softmax'))

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 2e-2, 3e-2, 4e-2, 5e-2])

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=['SparseCategoricalAccuracy'])

  return model

tuner = kt.Hyperband(model_builder,
                     objective='val_sparse_categorical_accuracy',
                     max_epochs=20,
                     factor=2,
                     directory='my_dir',
                     project_name='intro_to_kt')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

tuner.search(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the second densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train, epochs=100, validation_split=0.2)

val_acc_per_epoch = history.history['val_sparse_categorical_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(x_train, y_train, epochs=best_epoch, validation_split=0.2)