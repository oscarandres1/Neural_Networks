#|-- Load and plot dataset
import time
from datetime import datetime
import math
import pandas as pd
import numpy as np
import simplejson as json
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
import keras
import tensorflow.python.keras.backend as K
from keras import backend
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten, LSTM, Input, Concatenate, Dropout, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import plot_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
	
# fix random seed for reproducibility
np.random.seed(7)

PASOS=24

#| --convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True): 
    n_vars = 1 if type(data) is list else data.shape[1]   
    df = pd.DataFrame(data)      
    cols, names = list(), list()  
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):  
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)] #una sola variable 
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)] 
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

#| -- Cargar dataset
df = pd.read_csv('datosn.csv')
# cargar datos como fechas y float
df['Meses'] = pd.to_datetime(df['Meses'])
df['Exportaciones'] = df['Exportaciones'].astype('float32')
df.set_index('Meses', inplace=True)
#print(df.head(10))

# | -- Cargar dataset
values = df.values
#print(values.shape)

# |-- Normalizar features
scaler = MinMaxScaler(feature_range=(0, 1))
values = values.reshape(-1, 1) 
scaled = scaler.fit_transform(values)
# |-- frame as supervised learning
reframed = series_to_supervised(scaled, PASOS, 1)
print(reframed.shape)

# split into train and test sets
values = reframed.values
n_train_month = reframed.shape[0] - (12*4) 
train = values[:n_train_month-12, :] # [inidice fila: indice fila , indice col: indice col
valid = values[n_train_month-12:n_train_month, :] 
test = values[n_train_month:, :]

# split into input and outputs
x_train, y_train = train[:, :-1], train[:, -1]
x_val, y_val = valid[:, :-1], valid[:, -1]
x_test, y_test = test[:, :-1], test[:, -1]
#print(x_train)

# reshape input to be 3D [samples, timesteps, features]
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1],1))
x_val = x_val.reshape((x_val.shape[0], x_val.shape[1],1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1],1))
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)

# RMSE metric 
def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
 
# Hidden and output nuerons for 1 hidden layer network after CNN Layers
HIDDEN_NEURONS = 7
OUT_NEURONS = 1

def crear_modeloFF():
    inputs1 = Input(shape=(PASOS, 1), name='Input_1')
    cnn1D_1 = Conv1D(filters=4, kernel_size=7, strides=1, activation='relu', padding="same")(inputs1) 
    maxPl1D_1 = MaxPooling1D(pool_size=2, strides=1)(cnn1D_1)
    drop1 = Dropout(0.1)(maxPl1D_1)
    cnn1D_2 = Conv1D(filters=8, kernel_size=5, strides=1, activation='relu', padding="same")(drop1) 
    maxPl1D_2 = MaxPooling1D(pool_size=2, strides=2)(cnn1D_2)
    drop2 = Dropout(0.1)(maxPl1D_2)
    cnn1D_3 = Conv1D(filters=16, kernel_size=3, strides=1, activation='relu', padding="same")(drop2) 
    maxPl1D_3 = MaxPooling1D(pool_size=2, strides=2)(cnn1D_3)
    drop3 = Dropout(0.1)(maxPl1D_3)
    #cnn1D_4 = Conv1D(filters=32, kernel_size=1, strides=1, activation='relu', padding="same")(drop3) 
    #maxPl1D_4 = MaxPooling1D(pool_size=1, strides=2)(cnn1D_4)
    flattn = Flatten()(drop3)
    layer1 = Dense(HIDDEN_NEURONS, activation='relu')(flattn)
    drop4 = Dropout(0.1)(layer1)
    output1 = Dense(OUT_NEURONS, name='Dense_1')(drop4)
    model = Model(inputs=inputs1, outputs=output1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name="Adam")
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics = [rmse])
    plot_model(model, to_file='CNN', show_shapes=True)
    model.summary()
    return model

#----------------------------------------
EPOCHS = 800
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
  model = crear_modeloFF()
history = model.fit(x_train, y_train,validation_data=(x_val,y_val), epochs=EPOCHS, steps_per_epoch=5, batch_size=32, verbose=1)

scoretrain = model.evaluate(x_train, y_train, verbose = 0) 
scoretest = model.evaluate(x_test, y_test, verbose = 0) 

print('Train loss:', scoretrain[0]) 
print('Train rmse', scoretrain[1])
print('Test loss:', scoretest[0]) 
print('Test rmse:', scoretest[1])

# Predict and plot test data
results=model.predict(x_test)
#plt.scatter(range(len(y_val)),y_val)
#plt.scatter(range(len(results)),results)
#plt.title('Validate')
#plt.show()

# Predict and plot validation train
results1=model.predict(x_train)
#plt.plot(range(len(y_train)),y_train)
#plt.plot(range(len(results1)),results1)
#plt.title('Train')
#plt.show()

# Plot loss and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Vs RMSE Loss')
plt.show()

#Save and export
hist_df = pd.DataFrame(history.history) 
hist_df.to_csv('historyCNN.csv')
#----------------------------------------
#plt.title('Accuracy')
#plt.plot(history.history['mse'])
#plt.show()

#Invert and compare values
compara = pd.DataFrame(np.array([y_train, [x[0] for x in results1]])).transpose()
compara.columns = ['Real', 'Prediccion']
inverted = scaler.inverse_transform(compara.values)
realValues = pd.DataFrame(inverted)
realValues.columns = ['Real', 'Prediccion']
realValues.to_csv('entrenamientoCNN.csv')
realValues['Real'].plot(label = 'Real')
realValues['Prediccion'].plot(label = 'Predicted')
plt.legend()
plt.title('Train')
plt.show()

rmsetrain = math.sqrt(mean_squared_error(realValues['Real'], realValues['Prediccion']))
maetrain = mean_absolute_error(realValues['Real'],realValues['Prediccion'])
print('Train Score: %.2f RMSE' % (rmsetrain))
print('Train Score: %.2f MAE' % (maetrain))

compara = pd.DataFrame(np.array([y_test, [x[0] for x in results]])).transpose()
compara.columns = ['Real', 'Prediccion']
inverted = scaler.inverse_transform(compara.values)
realValues = pd.DataFrame(inverted)
realValues.columns = ['Real', 'Prediccion']
realValues.to_csv('testCNN.csv')
realValues['Real'].plot(label = 'Real')
realValues['Prediccion'].plot(label = 'Predicted')
plt.legend()
plt.title('Test')
plt.show()

rmsetest = math.sqrt(mean_squared_error(realValues['Real'], realValues['Prediccion']))
maetest = mean_absolute_error(realValues['Real'],realValues['Prediccion'])
print('Test Score: %.2f RMSE' % (rmsetest))
print('Test Score: %.2f MAE' % (maetest))

'''# |----------------------------------
ultimosMeses = df['01/08/2018':'01/08/2020']
# |----------------------------------
monthP = 12
values = ultimosMeses.values
values = values.astype(float)
# normalize features
values=values.reshape(-1, 1) 
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, PASOS, 1)
reframed.drop(reframed.columns[[monthP]], axis=1, inplace=True)
#print(reframed)

# |-------------------------------------------
values = reframed.values
#x_test = values[:, PASOS - 1:]
x_test = values[PASOS -1:, :] # agarramos la ultima fila perteneciente a los 12 ultimos meses 
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1],1)) # para el vector 1 x 12 x 12 [[[]]]
#print(x_test)

#-----------------------------------------
def addNewValue(x_test,nuevoValor):
    #print(x_test.shape[2] 1x12x1
    for i in range(x_test.shape[1]-1):
        x_test[0][i][0] = x_test[0][i+1][0] #DEsplaza a la izquierda los viejos valores
    x_test[0][x_test.shape[1]-1][0]=nuevoValor # coloca el nuevo valor predicho al final, para luego desplazarlo a la izquierda a medida que se colocan los nuevos valores predichos
    return x_test
 
results=[]
for i in range(12): # 12 meses a predecir 
    parcial=model.predict(x_test)
    results.append(parcial[0]) #parcial[0] es un valor 1x1
    #print(x_test)
    x_test=addNewValue(x_test,parcial[0])

#------------------------------------------
adimen = [x for x in results]    #para recuperar los valores 
inverted = scaler.inverse_transform(adimen)
#print(inverted)

#---------------------------------------
prediccionMeses = pd.DataFrame(inverted)
prediccionMeses.columns = ['pronostico']
prediccionMeses.plot(grid=True, figsize=(15,5))
plt.show()
prediccionMeses.to_csv('pronosticoCNN.csv')'''
	

