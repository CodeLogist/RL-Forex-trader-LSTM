import pandas as pd
import talib

df= pd.read_csv('data/train/EURUSD_H1_2010-2019_train.csv')

close = df['close'].astype('float')
volume = df['volume'].astype('float')
obv = talib.MA(close, volume)
print(obv)