from keras import Sequential
from keras.layers import Dense,Dropout
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from random import randint
import random
import numpy as np
from data import *
import keras
import knn
import forest
from joblib import dump, load

def arg_data(train_X,train_y,test_X):
    # from other methods get predict
    # and add to data
    ori_test_X = test_X
    knn_d = knn.predict(train_X,train_y,ori_test_X)
    knn_d = np.reshape(knn_d,(knn_d.shape[0],1))
    test_X = np.concatenate((test_X,knn_d),axis=1)
    forest_d = forest.predict(train_X,train_y,ori_test_X)
    forest_d = np.reshape(forest_d,(forest_d.shape[0],1))
    test_X = np.concatenate((test_X,forest_d),axis=1)
    return test_X

def acc_top10(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=10)

def get_smodel():
    # build model
    classifier = Sequential()
    #First Hidden Layer
    classifier.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform', input_dim=15,kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l1(0.01)))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l1(0.01)))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l1(0.01)))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(151, activation='softmax', kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l1(0.01)))
    classifier.compile(optimizer ='nadam',loss='sparse_categorical_crossentropy', metrics =[acc_top10])
    #classifier.load_weights('eq-model.h5')
    return classifier

if __name__ == '__main__':
    classifier = get_smodel()
    #classifier.compile(optimizer =optimizers.SGD(lr=1e-4),loss='binary_crossentropy', metrics =['accuracy'])
    filepath="eq-model.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    #
    for i in range(10):
        for j in range(10):
            if i == j:
                continue
            classifier.load_weights('eq-model.h5')
            X = load('X-{}.jl'.format(i))
            y = load('y-{}.jl'.format(i))
            train_X = load('X-{}.jl'.format(j))
            train_y = load('y-{}.jl'.format(j))
            classifier.fit(arg_data(train_X,train_y,X),y, epochs=10,callbacks=[checkpoint])