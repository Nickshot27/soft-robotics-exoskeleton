from pyomyo import Myo, emg_mode
import time
import multiprocessing
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

newModel = tf.keras.models.load_model('pyomyo_model_2.h5')


def modelProcessing(data_frame):
    maxChannels = data_frame.max()
    newMotion = np.array(maxChannels)
    newMotion = newMotion/1669.0
    newMotion = np.expand_dims(newMotion, 0)
    dist = newModel.predict(newMotion)
    predLabel = np.argmax(dist)
    return predLabel

def data_worker(mode, seconds):
    collect = True

    # ------------ Myo Setup ---------------
    m = Myo(mode=mode)
    m.connect()

    myo_data = []

    def add_to_queue(emg, movement):
        myo_data.append(emg)

    m.add_emg_handler(add_to_queue)

    def print_battery(bat):
        print("Battery level:", bat)

    m.add_battery_handler(print_battery)

    # Its go time
    m.set_leds([0, 128, 0], [0, 128, 0])
    # Vibrate to know we connected okay
    m.vibrate(1)

    print("Data Worker started to collect")
    # Start collecting data.
    start_time = time.time()

    for val in range(5):
        while collect:
            print("Collecting...")
            if (time.time() - start_time < seconds):
                m.run()
            else:
                collect = False
                myo_df = pd.DataFrame(myo_data)
                print("Iteration Done")
        moveClass = modelProcessing(myo_df)
        myo_data = []
        print(moveClass)
        collect = True
        start_time = time.time()


if __name__ == '__main__':

    seconds = 5
    mode = emg_mode.PREPROCESSED
    p = multiprocessing.Process(target=data_worker, args=(mode, seconds))
    p.start()

