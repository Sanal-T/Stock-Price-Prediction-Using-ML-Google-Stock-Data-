import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

df=pd.read_csv('GOOG.csv')

df.info()

df.shape
df=df.drop(columns=['symbol','adjClose','adjHigh','adjLow','adjOpen','adjOpen','divCash','splitFactor'],axis=1)
df.head()

df.describe()

import plotly.graph_objects as go
figure=go.Figure(data=[go.Candlestick(x=df['date'],open=df['open'],high=df['high'],low=df['low'],close=df['close'])])
figure.update_layout(title='GOOG',xaxis_rangeslider_visible=False)
figure.show()

df=df.drop(columns=['date'],axis=1)
df.head()

df.duplicated().sum().any()
df.isnull().values.any()
df.describe()

print(df.corr())

plt.figure(figsize=(16,8))
sns.heatmap(df.corr(),cmap='Blues',annot=True)
plt.show()

sns.pairplot(df)

df['open'].hist()

df['close'].hist()