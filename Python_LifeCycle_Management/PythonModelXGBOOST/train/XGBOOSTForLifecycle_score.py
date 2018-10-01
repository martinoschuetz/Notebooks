
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import xgboost
from sklearn.base import TransformerMixin
from sklearn.tree import DecisionTreeClassifier

from sklearn import model_selection
from sklearn.metrics import accuracy_score


# In[4]:


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


# In[6]:


df = pd.read_csv("/home/sasdemo01/PythonModel/hmeq_score.csv",na_values=['NULL'])
targetvar = ['BAD']
inputvars = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC']
Y = df[targetvar]
X0 = df[inputvars]
# impute the missing value
X = DataFrameImputer().fit_transform(X0)


# In[8]:


import pickle
loaded_model = pickle.load(open("/home/sasdemo01/PythonModelXGBOOST/train/xgboost_model_obj.pkl", "rb"))


# In[9]:


# build a xgboost model

loaded_model.fit(X,Y)

y_pred=loaded_model.predict(X)

y_pred

