from Scripts.Get_Logs import Get_Logs_LSTM as get_logs

import numpy as np
import random

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Bidirectional, LSTM, Dropout, Input
from keras import regularizers
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping,  ReduceLROnPlateau
from keras.preprocessing import sequence

def get_model(maxlen, num_chars, bidirec, n_layers, lstmsize, dropout, l1, l2):
    model = Sequential()
    model.add(Input(shape=(maxlen, num_chars))) #If you don't use an embedding layer input should be one-hot-encoded
    if bidirec == False:   
        model.add(LSTM(lstmsize,kernel_initializer='glorot_uniform',return_sequences=(n_layers != 1),
                       kernel_regularizer=regularizers.l1_l2(l1,l2), recurrent_regularizer=regularizers.l1_l2(l1,l2),
                       input_shape=(maxlen, num_chars)))
        model.add(Dropout(dropout))
        for i in range(1, n_layers):
            return_sequences = (i+1 != n_layers)
            model.add(LSTM(lstmsize,kernel_initializer='glorot_uniform',return_sequences=return_sequences,
            kernel_regularizer=regularizers.l1_l2(l1,l2),recurrent_regularizer=regularizers.l1_l2(l1,l2)))
            model.add(Dropout(dropout))
    else:
        model.add(Bidirectional(LSTM(lstmsize,kernel_initializer='glorot_uniform',return_sequences=(n_layers != 1),
                                     kernel_regularizer=regularizers.l1_l2(l1,l2), recurrent_regularizer=regularizers.l1_l2(l1,l2),
                                     input_shape=(maxlen, num_chars))))
        model.add(Dropout(dropout))
        for i in range(1, n_layers):
            return_sequences = (i+1 != n_layers)
            model.add(Bidirectional(LSTM(lstmsize,kernel_initializer='glorot_uniform',return_sequences=return_sequences, 
                                         kernel_regularizer=regularizers.l1_l2(l1,l2),recurrent_regularizer=regularizers.l1_l2(l1,l2))))
            model.add(Dropout(dropout))
    model.add(Dense(1, kernel_initializer='glorot_uniform',activation='sigmoid'))
    opt = Adam(learning_rate=0.005)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics='accuracy')
    return model

def get_integer_map(voc):
    return {x: i+1 for i,x in enumerate(voc)}

def apply_integer_map(log, map):
    return [[map[a] for a in t] for t in log]

def create_train(log, antilog, vocsize, maxlen): #we one-hot-encode because we don't use embedding layer
    XY = []
    for i in range(0,len(log)):
        trace = []
        for l in range(0, len(log[i])):
            onehot = [0]*vocsize
            onehot[log[i][l] - 1] = 1
            trace.append(onehot)
        XY.append([trace, [1]])
    for j in range(0,len(antilog)):
        trace = []
        for m in range(0, len(antilog[j])):
            onehot = [0]*vocsize
            onehot[antilog[j][m] - 1] = 1
            trace.append(onehot)
        XY.append([trace, [0]])
    random.shuffle(XY)
    X = []
    Y = []
    for k in range(0, len(XY)):
        X.append(XY[k][0])
        Y.append(XY[k][1])     
    X = sequence.pad_sequences(X, maxlen=maxlen)
    return np.array(X),np.array(Y)

def create_test(log, vocsize, maxlen):
    X = []
    for i in range(0,len(log)):
        trace = []
        for l in range(0, len(log[i])):
            onehot = [0]*vocsize
            onehot[log[i][l] - 1] = 1
            trace.append(onehot)
        X.append(trace)
    X = sequence.pad_sequences(X, maxlen=maxlen)
    return np.array(X)  

def  get_dist_random(filenamelog, filenamemodel, modellogsize=None, antilogsize=None, modelantilogsize=None, max_occ=3, bidirec=False, n_layers=1, lstmsize=16, dropout=0.0, l1=0.0, l2=0.0, batch_size=64): 
    event_log = get_logs.get_event_log(filenamelog, token=False)
    if modellogsize == None:
        modellogsize = len(event_log)
    if antilogsize == None:
        antilogsize = len(event_log)
    if modelantilogsize == None:
        modelantilogsize = len(event_log)
        
    model_log = get_logs.get_model_log(filenamemodel, logsize=modellogsize, maxtracelength=100, mintracelength=1, token=False, max_occ = max_occ)
    
    voc = get_logs.get_voc(event_log+model_log)
    
    event_antilog = get_logs.get_antilog_random(event_log, voc, log_size=antilogsize, delete_correct=False)
    model_antilog = get_logs.get_antilog_random(model_log, voc, log_size=modelantilogsize, delete_correct=False)
    
    maxlen = len(max(model_log+event_log, key=len))
    
    mapping = get_integer_map(voc)
    
    event_log = apply_integer_map(event_log, mapping)
    model_log = apply_integer_map(model_log, mapping)
    event_antilog = apply_integer_map(event_antilog, mapping)
    model_antilog = apply_integer_map(model_antilog, mapping)
    
    #first we do the fitness calculations   
    X_train_fitness, y_train_fitness = create_train(model_log, model_antilog,len(voc), maxlen)
    X_test_fitness = create_test(event_log, len(voc), maxlen)
    

    model = get_model(maxlen, len(voc), bidirec, n_layers, lstmsize, dropout, l1, l2)
    model.summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=9, restore_best_weights=True)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    #train_model, this was changed to use test log as validation (because it is here)
    history = model.fit(X_train_fitness, y_train_fitness,validation_split = 0.2, callbacks=[early_stopping, lr_reducer], batch_size=batch_size, epochs=100, verbose=0)

    fitness = np.average(model.predict(X_test_fitness))
    
    #first we do the fitness calculations
    X_train_precision, y_train_precision = create_train(event_log, event_antilog,len(voc), maxlen)
    X_test_precision = create_test(model_log, len(voc), maxlen)
    model2 = get_model(maxlen, len(voc), bidirec, n_layers, lstmsize, dropout, l1, l2)
    model2.summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=9, restore_best_weights=True)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    #train_model, this was changed to use test log as validation (because it is here)
    history2 = model2.fit( X_train_precision, y_train_precision,validation_split = 0.2, callbacks=[early_stopping, lr_reducer], batch_size=batch_size, epochs=100, verbose=0)

    precision = np.average(model2.predict(X_test_precision))
    
    return(fitness, precision)