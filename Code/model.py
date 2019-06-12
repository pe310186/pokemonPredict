from keras import Sequential
from keras.layers import Dense,Dropout
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from random import randint
import numpy as np
from data import *
import keras

def get_model():
    # build model
    classifier = Sequential()
    #First Hidden Layer
    classifier.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform', input_dim=13,kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l1(0.01)))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l1(0.01)))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l1(0.01)))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(144, activation='softmax', kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l1(0.01)))
    #classifier.compile(optimizer ='nadam',loss='categorical_crossentropy', metrics =[keras.metrics.top_k_categorical_accuracy])
    #classifier.load_weights('w.h5')
    return classifier

if __name__ == '__main__':
    classifier = get_model()
    filepath="w.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    #classifier.load_weights(filepath)
    X,y = get_data()
    classifier.fit(X,y,batch_size=5, epochs=2,callbacks=[checkpoint], validation_split=0.3)