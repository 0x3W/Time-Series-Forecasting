import subprocess
import numpy as np
from sklearn import linear_model
import pandas as pd
import json  
import pickle
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import statsmodels.stats as sms
import scipy.stats as scs
from statsmodels.tsa.api import ExponentialSmoothing
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, TimeDistributed, Dropout

def process(smth, mod,modScale,sgdScale,ForMod):
    df = pd.DataFrame()
    targ = smth['target']['name']
    df[targ] = smth['target']['data']
    for i in range(len(smth['features'])):
        col = smth['features'][i]['name']
        df[col] = smth['features'][i]['data']
    uniPred, multiPred, naive, ma, mod, hold = regr(df,mod,modScale,sgdScale,ForMod)
    return mod, { 'timestamp': smth['timestamp'], 'target': targ, 'unPred': unPred, 'multiPred': multPred,'Naive':naive,'MA':ma}

def regr(df,mod,modScale,sgdScale=1,ForMod=1):
    hold = 0
    sgd = linear_model.SGDRegressor(max_iter=1000,alpha=0.0001, penalty='elasticnet')
    if sgdScale==1:
        scaler = StandardScaler()
        normalized = scaler.fit_transform(df.iloc[:,1:9])
        xTs = pd.DataFrame(normalized)
        sgd.fit(pd.DataFrame(normalized),df.iloc[:,0])
        if ForMod == 2:
            predDat, uniPred,arimaPred = arimaRNN(df,mod)
        if ForMod == 1:
            predDat, uniPred = rscript(df)
        else:
            predDat, uniPred, mod = esRNN(df,mod,modScale=0)
            if len(predDat) == 1:
                norm = scaler.fit_transform(np.array(predDat).reshape(-1,1))
            else:
                norm = scaler.fit_transform(np.array(predDat).reshape(1,-1))
        print(norm)
        multiPred = sgd.predict(norm)
    else:
        sgd.fit(df.iloc[:,1:9],df.iloc[:,0])
        if ForMod == 2:
            predDat, uniPred,arimaPred = arimaRNN(df,mod)
            hold = arimaPred
        if ForMod == 1:
            predDat, uniPred = rscript(df)
        else:
            predDat, uniPred, mod = esRNN(df,mod,modScale=0)
        multiPred = sgd.predict([np.asarray(predDat)])
    naive = df.iloc[:,0].mean()
    m = df.iloc[:,0].rolling(2).mean()
    ma = m[m.shape[0]-1]
    return uniPred, multiPred.item(0), naive, ma, mod, hold

def rscript(df):
    newPreds = []
    lags = range(1, 10)
    for i in range(min(len(df.columns),9)):
        x1 = pd.DataFrame(df.iloc[:,i])
        x1.to_csv('test3.csv')
        process = subprocess.Popen(['Rscript','try.R','test3.csv'], stdout=subprocess.PIPE)
        process.wait()
        stdout = process.communicate()[0].decode('ascii')
        stdout1 = float(stdout.split(' ',2)[1].split("\n",-1)[0])
        newPreds.append(stdout1)
    u = newPreds.pop(0)
    return newPreds, u

def genNet():
    model = Sequential()
    model.add(LSTM(16, activation='relu', input_shape=(8, 1)))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def esRNN(df,mod,modScale=0):
    newPreds = []
    lags = range(1, 10)
    for i in range(min(len(df.columns),9)):
        xTs = df.iloc[:,i]
        fit1 = ExponentialSmoothing(xTs, seasonal_periods=100, trend='add', seasonal='add').fit(use_boxcox=False)
        x1 =  pd.DataFrame(xTs / (fit1.level * fit1.season))
        x2 = x1.assign(**{
                '{} (t-{})'.format(col, t): x1[col].shift(t)
                for t in lags
                for col in x1
        })
        x3 = x2.dropna(axis=0)
        Y = np.array(x3.iloc[:,0])
        X = np.array(x3.iloc[:,1:9].fillna(0))
        if modScale == 1:
            scaler = StandardScaler()
            scaler = scaler.fit(X)
            X = scaler.transform(X)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        if mod != 0:
            model = mod
            model.fit(X,Y)
            mod = model
            print('existsDeep')
        else:
            model = genNet()
            model.fit(X, Y, epochs=10, verbose=0)
            mod = model
        x4 = np.asarray(x3.iloc[-1])
        if modScale == 1:
            scaler = StandardScaler()
            scaler = scaler.fit(x4)
            x4 = scaler.transform(x4)
            x5 = x4[0:8].reshape((1,8,1))
            yh = model.predict(x5, verbose=0)
            yhat = scaler.inverse_transform(yh.item(0))
        else:
            x5 = x4[0:8].reshape((1,8,1))
            yhat = model.predict(x5, verbose=0)
        yhat2 = yhat.item(0) * fit1.level.iloc[-1] * fit1.season.iloc[-1]
        newPreds.append(yhat2)
    u = newPreds.pop(0)
    return newPreds, u, mod

def bestArMod(x):
    aic = np.inf 
    order = None
    mdl = None
    pq = range(3)
    for i in pq:
        for j in pq:
            try:
                tempMdl = ARIMA(x, order=(i,0,j)).fit(method='mle', trend='nc', disp=0)
                tmpAic = tmpMdl.aic
                if tmp_aic < aic:
                    aic = tmpAic
                    order = (i, 0, j)
                    mdl = tmpMdl
            except: continue
    return mdl   

def arimaRNN(df,mod):
    newPreds = []
    lags = range(1, 10)
    print(df.columns)
    for i in range(min(len(df.columns),9)):
        xTs = df.iloc[:,i]
        modelAr = bestArMod(xTs)
        yhat = modelAr.forecast()[0]
        x1 = pd.DataFrame(modelAr.resid)       
        x2 = x1.assign(**{
                '{} (t-{})'.format(col, t): x1[col].shift(t)
                for t in lags
                for col in x1
        })
        x3 = x2.dropna(axis=0)
        Y = np.array(x3.iloc[:,0])
        X = np.array(x3.iloc[:,1:].fillna(0))
        X = X.reshape((X.shape[0], X.shape[1], 1))
        if mod != 0:
            model = mod
            model.fit(X,Y)
            mod = model
        else:
            model = genNet()
            model.fit(X, Y, epochs=10, verbose=0)
            mod = model
        x4 = np.asarray(x3.iloc[-1])
        x5 = x4[0:8].reshape((1,8,1))
        yhatRNN = model.predict(x5, verbose=0)
        yhat2 = yhatRNN.item(0)
        newPreds.append(yhat2+yhat)
    u = newPreds.pop(0)
    return newPreds, u, mod, yhat
