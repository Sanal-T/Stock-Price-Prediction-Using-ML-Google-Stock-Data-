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