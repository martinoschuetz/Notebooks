
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import xgboost
from sklearn.base import TransformerMixin
from sklearn.tree import DecisionTreeClassifier

from sklearn import model_selection
from sklearn.metrics import accuracy_score


# In[2]:


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


# In[3]:


df = pd.read_csv("/home/sasdemo01/PythonModel/hmeq.csv",na_values=['NULL'])
targetvar = ['BAD']
inputvars = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC']
Y = df[targetvar]
X0 = df[inputvars]
# impute the missing value
X = DataFrameImputer().fit_transform(X0)
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
test_size = 0.33
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=0)


# In[4]:


# build a xgboost model
model = xgboost.XGBClassifier()
model.fit(X_train, y_train)

y_pred=model.predict(X_test)

print("accuracy on training set: %f" % model.score(X_train, y_train))
print("accuracy on test set: %f" % model.score(X_test, y_test))
y_pred


# In[5]:


#export the model to a external file for later use
import pickle 
pklfile = open("/home/sasdemo01/PythonModelXGBOOST/train/xgboost_model_obj.pkl",'wb')
pickle.dump(model, pklfile)
pklfile.close()



