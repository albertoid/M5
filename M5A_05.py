import pandas as pd
import numpy as np
from tqdm import tqdm as tqdm
import pickle

#Models
from sklearn.model_selection import train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
import xgboost as xgb
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK
from hyperopt.pyll import scope as ho_scope
from hyperopt.pyll.stochastic import sample as ho_sample

import warnings
warnings.filterwarnings('ignore')

print('Cargando Bases de Datos')
df_cal=pd.read_csv('a/calendar.csv')
df_sat=pd.read_csv('a/sales_train_validation.csv')
df_sam=pd.read_csv('a/sample_submission.csv')
df_sep=pd.read_csv('a/sell_prices.csv')

#Loading Pickle
print('Cargando Datos Preprocesados')
file = open('list_df_sell.pickle', 'rb')
list_df_sell = pickle.load(file)
file.close()

space={
    'n_estimators':hp.quniform('n_estimators', 10, 2000, 25),
    'learning_rate':hp.uniform('learning_rate', 0.00001, 1.0),
    'max_depth':hp.quniform('x_max_depth', 8, 32, 1),
    'min_child_weight':hp.quniform('x_min_child', 1, 10, 1),
    'subsample':hp.uniform('x_subsample', 0.7, 1),
    'gamma':hp.uniform('x_gamma', 0.1, 0.5),
    'reg_lambda':hp.uniform('x_reg_lambda', 0, 1)
}

def xgbmodel(i,ShowMSE=False, max_evals =5):
    trials_reg=Trials()

    def objetivo(space):
        modelo=xgb.XGBRegressor(
            n_estimators=int(space['n_estimators']),
            learning_rate=space['learning_rate'],
            max_depth=int(space['max_depth']),
            min_child_weight=space['min_child_weight'],
            subsample=space['subsample'],
            gamma=space['gamma'],
            reg_lambda=space['reg_lambda'],
            objective='reg:squarederror'
        )
        
        eval_set=[(X_train, y_train), (X_test, y_test)]
        modelo.fit(X_train, y_train, eval_set=eval_set, eval_metric='rmse', verbose=False)
        y_pred=modelo.predict(X_test)
        rmse=MSE(y_test, y_pred)**0.5
        return {'loss':rmse, 'status':STATUS_OK}

    Xy=list_df_sell[i].copy()

    #Stracting X_test
    X_test=Xy.drop(columns=['day_sell'])[1913:]

    #Remove Nones for all the sample  from X and y in the sell price
    Xy.sell_price=Xy.sell_price.astype(float)
    Xy=Xy[(Xy.sell_price>0) & (Xy.day_sell >=0)]

    #Prepare X_train and y_train
    X,y = Xy.drop(columns=['day_sell']), Xy.day_sell
    

    X_train, X_test, y_train, y_test = TTS(X,y, test_size = 0.2, shuffle=False)
    best=fmin(fn=objetivo, space=space, algo=tpe.suggest, max_evals=max_evals, trials=Trials())

    #Train with complete data set and founded hyperparameters

    modelo=xgb.XGBRegressor(
        n_estimators=int(best['n_estimators']),
        learning_rate=best['learning_rate'],
        x_max_depth=int(best['x_max_depth']),
        x_min_child=best['x_min_child'],
        x_subsample=best['x_subsample'],
        x_gamma=best['x_gamma'],
        x_reg_lambda=best['x_reg_lambda'],
        objective='reg:squarederror'
        )

    #Checking MSE
    if ShowMSE==True:
        modelo.fit(X_train, y_train)
        y_pred=modelo.predict(X_test)
        print(MSE(y_test, y_pred))

    
    #Defining new X_train and y_train to train with the all dataset
    Xy=list_df_sell[i].copy()

    #Stracting X_test
    X_test=Xy.drop(columns=['day_sell'])[1913:]
    X_test.sell_price=X_test.sell_price.astype(float)

    #Remove Nones for all the sample  from X and y in the sell price
    Xy.sell_price=Xy.sell_price.astype(float)
    Xy=Xy[(Xy.sell_price>0) & (Xy.day_sell >=0)]

    #Prepare X_train and y_train
    X,y = Xy.drop(columns=['day_sell']), Xy.day_sell
    
    
    #Final Train
    m=xgb.XGBRegressor()
    m.fit(X_train,y_train)
    y_pred=m.predict(X_test)
    
    y_pred=np.array(list((map(lambda x: 0 if x<0 else x,y_pred))))
    
    return y_pred


def prediction(i, database_size=30490):
    
    #Applying model

    y_pred=xgbmodel(i)

    #Write predictions in sumbit validation (first 28)
    df_sub.iloc[i,1:] = y_pred[:28]

    #Write predictions in submit validation (second 28) row + 30490
    df_sub.iloc[i+database_size,1:]=y_pred[28:]

#MAIN RUN
df_sub=df_sam.copy()

for i in tqdm(range(10)):
#for i in tqdm(range(len(df_sat))):
    prediction(i)

df_sub.to_csv('M5_AV_05_XGBoost_n5.csv', index=False)