import pandas as pd
import numpy as np
from tqdm import tqdm as tqdm
import pickle
import multiprocessing as mp

#Modelos
from sklearn.ensemble import RandomForestRegressor as RFR

import warnings
warnings.filterwarnings('ignore')

print('Cargando Archivos Fuente')

df_cal=pd.read_csv('a/calendar.csv')
df_sat=pd.read_csv('a/sales_train_validation.csv')
df_sam=pd.read_csv('a/sample_submission.csv')
df_sep=pd.read_csv('a/sell_prices.csv')

#Loading Pickle
print('Cargando Pickle')
file = open('list_df_sell.pickle', 'rb')
list_df_sell = pickle.load(file)
file.close()

def prediction(i, database_size=30490, rounded=True):
    Xy=list_df_sell[i].copy()
    
    #Stracting X_test
    X_test=Xy.drop(columns=['day_sell'])[1913:]
    
    #Remove Nones for all the sample  from X and y in the sell price
    Xy.sell_price.astype(float)
    Xy=Xy[(Xy.sell_price>0) & (Xy.day_sell >=0)]
    
    #Prepare X_train and y_train
    X_train,y_train = Xy.drop(columns=['day_sell']), Xy.day_sell
    
    #Applying model to 56 days
    rfr=RFR()
    rfr.fit(X_train,y_train)
    y_pred=rfr.predict(X_test)

    if rounded == True:
        #Write predictions in sumbit validation (first 28)
        df_sub.iloc[i,1:] = y_pred[:28].round()

        #Write predictions in submit validation (second 28) row + 30490
        df_sub.iloc[i+database_size,1:]=y_pred[28:].round()
    else:
        #Write predictions in sumbit validation (first 28)
        df_sub.iloc[i,1:] = y_pred[:28]

        #Write predictions in submit validation (second 28) row + 30490
        df_sub.iloc[i+database_size,1:]=y_pred[28:]
        
        
#Copy to the sample (df_sam) = submit (df_sub)
df_sub=df_sam.copy()

for i in tqdm(range(len(df_sat))):
    prediction(i,rounded=False)

print('Generando archivo csv')
df_sub.to_csv('M5_AV_02_Random_Forest_Non_Rounded.csv', index=False)